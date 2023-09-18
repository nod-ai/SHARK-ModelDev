# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...dynamo.importer import (
    ContextCache,
    TORCH_DTYPE_TO_MLIR_TYPE_ASM,
)

from .ir_imports import (
    BF16Type,
    ComplexType,
    DenseElementsAttr,
    F16Type,
    F32Type,
    F64Type,
    FunctionType,
    InsertionPoint,
    IntegerType,
    IrType,
    Location,
    Operation,
    RankedTensorType,
    StringAttr,
    SymbolTable,
    TypeAttr,
    UnitAttr,
    Value,
    func_d,
)

from .utils import (
    RefTracker,
)

###############################################################################
# Lookup tables
###############################################################################

# We need the inverse of the TORCH_DTYPE_TO_MLIR_TYPE_ASM table.
MLIR_TYPE_ASM_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_MLIR_TYPE_ASM.items()}

# When emitting constants, we have to create native IREE types.
TORCH_DTYPE_TO_IREE_TYPE: Dict[str, Callable[[], IrType]] = {
    torch.float16: lambda: F16Type.get(),
    torch.bfloat16: lambda: BF16Type.get(),
    torch.float32: lambda: F32Type.get(),
    torch.float64: lambda: F64Type.get(),
    torch.uint8: lambda: IntegerType.get_signless(8),
    torch.int8: lambda: IntegerType.get_signless(8),
    torch.int16: lambda: IntegerType.get_signless(16),
    torch.int32: lambda: IntegerType.get_signless(32),
    torch.int64: lambda: IntegerType.get_signless(64),
    torch.bool: lambda: IntegerType.get_signless(1),
    torch.qint8: lambda: IntegerType.get_signless(8),
    torch.quint8: lambda: IntegerType.get_signless(8),
    torch.complex32: lambda: ComplexType.get(F16Type.get()),
    torch.complex64: lambda: ComplexType.get(F32Type.get()),
    torch.complex128: lambda: ComplexType.get(F64Type.get()),
}

TORCH_DTYPE_TO_IREE_TYPE_ASM = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.uint8: "i8",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "i1",
    torch.qint8: "i8",
    torch.quint8: "i8",
    torch.complex32: "complex<f16>",
    torch.complex64: "complex<f32>",
    torch.complex128: "complex<f64>",
}

###############################################################################
# Builders
###############################################################################


class ModuleBuilder:
    """Wrapper around module and IR accounting for a module being built."""

    __slots__ = [
        "body",
        "cache",
        "context",
        "global_ip",
        "ip",
        "module_op",
        "symbol_table",
        "global_ref_tracker",
    ]

    def __init__(self, module_op: Operation):
        self.module_op = module_op
        self.context = module_op.context
        self.body = module_op.regions[0].blocks[0]
        self.symbol_table = SymbolTable(module_op)
        self.global_ip = InsertionPoint.at_block_begin(self.body)
        self.ip = InsertionPoint(self.body)
        self.cache = ContextCache(self.context)
        # Tracks global references to a MaterializedGlobal.
        self.global_ref_tracker = RefTracker()

    def finalize_construct(self):
        self.module_op.verify()

    def create_func_op(
        self,
        symbol_name: str,
        argument_types: Sequence[IrType],
        is_public: bool = True,
        add_entry_block: bool = True,
    ) -> Tuple[str, func_d.FuncOp]:
        with self.ip:
            ftype = FunctionType.get(argument_types, [])
            func_op = func_d.FuncOp(symbol_name, ftype)
            if not is_public:
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
            if add_entry_block:
                func_op.add_entry_block()
            self.symbol_table.insert(func_op)
            actual_symbol_name = StringAttr(func_op.attributes["sym_name"]).value
            return actual_symbol_name, func_op

    def torch_dtype_to_iree_type(self, dtype: torch.dtype) -> IrType:
        try:
            with self.context:
                return TORCH_DTYPE_TO_IREE_TYPE[dtype]()
        except KeyError:
            raise TypeError(f"Could not map Torch dtype {dtype} to an IREE type")

    def create_tensor_global(
        self,
        symbol_name: str,
        t: torch.Tensor,
        *,
        mutable: bool = False,
        initialize: bool = True,
        noinline: bool = True,
    ) -> Tuple[str, Operation, IrType]:
        element_type = self.torch_dtype_to_iree_type(t.dtype)
        with self.global_ip, Location.unknown():
            tensor_type = RankedTensorType.get(list(t.shape), element_type)
            attrs = {
                "sym_name": StringAttr.get(symbol_name),
                "sym_visibility": StringAttr.get("private"),
                "type": TypeAttr.get(tensor_type),
            }
            if noinline:
                attrs["noinline"] = UnitAttr.get()
            if mutable:
                attrs["is_mutable"] = UnitAttr.get()
            if initialize:
                detached_tensor = t.detach().contiguous().cpu()
                array = np.array(detached_tensor)
                contents = memoryview(array)
                # TODO: Add resource elements to Python API and use that.
                elements_attr = DenseElementsAttr.get(contents, type=tensor_type)
                attrs["initial_value"] = elements_attr

            global_op = Operation.create("util.global", attributes=attrs)
            self.symbol_table.insert(global_op)
            actual_symbol_name = StringAttr(global_op.attributes["sym_name"]).value
            return actual_symbol_name, global_op, tensor_type


class FunctionBuilder:
    """Helpers for building function bodies."""

    __slots__ = [
        "module_builder",
        "func_op",
        "context",
        "ip",
        "return_types",
        "loc",
    ]

    def __init__(
        self,
        *,
        module_builder: ModuleBuilder,
        func_op: func_d.FuncOp,
    ):
        self.module_builder = module_builder
        self.func_op = func_op
        self.context = func_op.context
        self.ip = InsertionPoint(self.func_op.entry_block)
        self.return_types = None
        self.loc = self.func_op.location

    def emit_return(self, *ir_values: Sequence[Value]):
        with self.loc, self.ip:
            func_d.ReturnOp(ir_values)
            # Check or rewrite the function return type.
            value_types = [v.type for v in ir_values]
            if self.return_types:
                if value_types != self.return_types:
                    raise ValueError(
                        f"Multi-return function must return same types. "
                        f"{value_types} vs {self.return_types}"
                    )
                return
            self.return_types = value_types
            ftype = self.func_op.type
            ftype = FunctionType.get(ftype.inputs, value_types)
            self.func_op.attributes["function_type"] = TypeAttr.get(ftype)
            assert self.func_op.verify(), "Created function is invalid"

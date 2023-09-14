# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Dict, Generator, Optional, Sequence, Tuple

import logging
import weakref

import numpy as np
import torch

from iree.compiler.ir import (
    Context,
    DenseElementsAttr,
    FunctionType,
    InsertionPoint,
    Location,
    Operation,
    StringAttr,
    SymbolTable,
    Type as IrType,
    TypeAttr,
    UnitAttr,
    # Types.
    ComplexType,
    BF16Type,
    F16Type,
    F32Type,
    F64Type,
    IntegerType,
    RankedTensorType,
)

from iree.compiler.dialects import (
    func as func_d,
    util as util_d,
)

from ..dynamo.importer import (
    ContextCache,
)

logger = logging.getLogger("shark_turbine.aot")

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


# Opaque value to indicate something is empty. Used in cases where 'None'
# may have a different meaning.
_Empty = object()


class RefMapping:
    __slots__ = [
        "_referrent",
        "value",
    ]

    def __init__(self, referrent: Any):
        if referrent is not _Empty:
            self._referrent = weakref.ref(referrent)
        self.value = _Empty

    @property
    def is_empty(self):
        return self.value is _Empty

    def __repr__(self):
        return (
            f"<RefMapping {id(self._referrent) if self._referrent is not _Empty else 'empty'} -> "
            f"{self.value if self.value is not _Empty else 'empty'}>"
        )


class RefTracker:
    """Tracks live references from Python values to symbolic associations."""

    def __init__(self):
        self._refs: Dict[int, RefMapping] = {}

    def track(self, referrent: Any) -> RefMapping:
        ref_id = id(referrent)
        existing = self._refs.get(ref_id)
        if existing:
            return existing
        info = RefMapping(referrent)
        if referrent is not _Empty:
            weakref.finalize(referrent, self._ref_finalizer, ref_id)
        self._refs[ref_id] = info
        return info

    def _ref_finalizer(self, ref_id: int):
        del self._refs[ref_id]


class GlobalsDef:
    """Base class for all exporting descriptors."""

    __slots__ = []

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """Yields tuples of name/value exports."""
        raise NotImplementedError


class MaterializedGlobal:
    """Associates a (possibly) materialized global with a name hint and info for the aggregate it is part of."""

    __slots__ = [
        "global_op",
        "global_type",
        "info",
        "export_name",
        "symbol_name",
    ]

    def __init__(
        self,
        export_name: str,
        info: GlobalsDef,
        *,
        symbol_name: str,
        global_op: Operation,
        global_type: IrType,
    ):
        self.info = info
        self.export_name = export_name
        self.symbol_name = symbol_name
        self.global_op = global_op
        self.global_type = global_type

    def __repr__(self):
        return f"<MaterializedGlobal {self.export_name} = {self.symbol_name}:{self.global_type}>"


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
        initial_value: bool = True,
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
            if initial_value:
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

    def track_globals(self, export_namespace: str, globals_def: GlobalsDef):
        for name, value in globals_def.items():
            # Switch on types we support.
            if isinstance(value, torch.Tensor):
                fq_name = f"{export_namespace}.{name}"
                mapping = self.global_ref_tracker.track(value)
                if not mapping.is_empty:
                    logging.debug(
                        "IGNORE EXISTING TRACKED TENSOR(%s): %r", fq_name, mapping
                    )
                    continue
                actual_symbol_name, global_op, global_type = self.create_tensor_global(
                    f"_{fq_name}", value
                )
                mapping.value = MaterializedGlobal(
                    fq_name,
                    globals_def,
                    symbol_name=actual_symbol_name,
                    global_op=global_op,
                    global_type=global_type,
                )
                # TODO: Handle GlobalsDef that request non-lazy instantiation.
                logging.debug("TRACK NEW TENSOR(%s): %r", fq_name, mapping)
                continue

            raise TypeError(f"Unsupported global type: {value.__class__}")

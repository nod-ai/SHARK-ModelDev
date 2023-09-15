# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

from contextlib import contextmanager
import logging
import threading
import weakref

import numpy as np
import torch

from torch.utils._pytree import (
    TreeSpec,
    tree_flatten,
    tree_map,
    tree_unflatten,
    treespec_dumps,
)

from iree.compiler.ir import (
    Context,
    DenseElementsAttr,
    FunctionType,
    InsertionPoint,
    Location,
    Operation,
    RankedTensorType,
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
    Value,
)

from iree.compiler.dialects import (
    func as func_d,
    util as util_d,
)

from ..dynamo.importer import ContextCache

logger = logging.getLogger("shark_turbine.aot")

_thread_state = threading.local()

###############################################################################
# Lookup tables
###############################################################################

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
# Reference mapping
###############################################################################

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


###############################################################################
# Builders
###############################################################################

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


###############################################################################
# Tracing intrinsics
###############################################################################


class IrTrace(FunctionBuilder):
    """Gets callbacks for tracing events."""

    __slots__ = []

    def finalize(self):
        """Called when the trace is finished (popped off the stack)."""
        pass

    def handle_call(self, target: "Intrinsic", args, kwargs):
        raise NotImplementedError(f"The current trace scope does not support calls")

    def handle_assignment(self, scope, target, updated_value):
        raise NotImplementedError(
            f"The current trace scope does not support assignment"
        )


def _trace_scopes() -> List[IrTrace]:
    try:
        trace_scopes = _thread_state.trace_scopes
    except AttributeError:
        trace_scopes = _thread_state.trace_scopes = []
    return trace_scopes


@contextmanager
def new_ir_trace_scope(ir_trace: IrTrace):
    trace_scopes = _trace_scopes()
    trace_scopes.append(ir_trace)
    try:
        yield ir_trace
    finally:
        ir_trace.finalize()
        del trace_scopes[-1]


def current_ir_trace() -> IrTrace:
    return _trace_scopes()[-1]


class Intrinsic:
    """Objects which interact natively with the tracing system implement this."""

    __slots__ = []

    def resolve_ir_values(self, proc_trace: "IrTrace") -> Sequence[Value]:
        raise NotImplementedError(
            f"Cannot use {self} as an expression in a procedural function"
        )

    def resolve_call(self, proc_trace: "IrTrace", *args, **kwargs):
        raise NotImplementedError(
            f"Cannot use {self} as the target of a call in a procedural function"
        )

    def resolve_assignment(self, proc_trace: "IrTrace", ir_values: Sequence[Value]):
        raise NotImplementedError(
            f"Cannot use {self} as the target of an assignment in a procedural function"
        )


class CallableIntrinsic(Intrinsic):
    """Intrinsic subclass that supports calls.

    This is separate so as to make error handling better (i.e. does not support
    calls) for intrinsics that are not callable.
    """

    __slots__ = []

    def __call__(self, *args, **kwargs):
        return current_ir_trace().handle_call(self, args, kwargs)


class AbstractIntrinsic:
    """Base class for descriptor types that can be converted to Python proxies."""

    __slots__ = []

    def create_intrinsic(self, value: Value) -> Intrinsic:
        """Creates a proxy object that can flow through a procedural trace."""
        raise NotImplementedError

    def get_ir_type(self, builder: ModuleBuilder) -> IrType:
        """Gets the corresponding IR type."""
        raise NotImplementedError


###############################################################################
# Tensors
###############################################################################


class IrTensorBase(Intrinsic):
    """Base class for 'tensors' that resolve to some IR value.

    These are not real tensors (i.e. no operations can be performed on them),
    but they stand in as reasonable proxies during procedural tracing.
    """

    __slots__ = [
        "ir_type",
        "dtype",
    ]

    def __init__(self, ir_type: IrType, dtype: torch.dtype):
        self.ir_type = ir_type
        self.dtype = dtype

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return NotImplemented

    def _to_meta_tensor(self) -> torch.Tensor:
        """Converts to a fake Tensor that dynamo can handle."""
        ir_tensor_type = RankedTensorType(self.ir_type)
        shape = ir_tensor_type.shape
        # TODO: Remove this assert. We need to extend this method to also be
        # able to contribute dynamic_dim constraints and then return a minimum
        # quantity (2) for any dynamic dim.
        assert not any(
            d < 0 for d in shape
        ), "Dynamic dims to jittable not yet implemented"

        # TODO: We shouldn't need to create a real tensor here, as Dynamo will
        # immediately convert it to fake. However, it will also set up the shape
        # environment and asserts that any fake tensor inputs are from its
        # internal FakeMode. There should be a way but needs more investigation.
        # TODO: This tensor needs a device that matches the model being exported.
        # We just create these on the CPU because that is common.
        return torch.empty(shape, dtype=self.dtype)


class IrValueTensor(IrTensorBase):
    """Represents a Value in the IR under construction during procedural tracing."""

    __slots__ = [
        "ir_value",
    ]

    def __init__(self, ir_value: Value, dtype: torch.dtype):
        super().__init__(ir_value.type, dtype)
        self.ir_value = ir_value

    def __repr__(self):
        return f"IrValueTensor(@{self.ir_value})"

    def resolve_ir_values(self, proc_trace: IrTrace) -> Sequence[Value]:
        return (self.ir_value,)


###############################################################################
# Abstract types
###############################################################################


class AbstractTypedef:
    """Base class for instances which declare some form of public arg/result type definition."""

    ...


class Abstractifiable:
    """Indicates that a type knows how to abstractify itself."""

    def abstractify(self) -> AbstractTypedef:
        raise NotImplementedError


class TreeAbstractifiable:
    """Indicates that a type decomposes into a tree that can be abstractified."""

    def abstractify_tree(self) -> Any:
        raise NotImplementedError


class AbstractTensor(AbstractIntrinsic, AbstractTypedef):
    """Represents a tensor of known rank and dtype."""

    __slots__ = [
        "size",
        "dtype",
    ]

    def __init__(self, *size: Optional[int], dtype: torch.dtype = torch.float32):
        self.size = tuple(size)
        self.dtype = dtype

    def __repr__(self):
        return f"AbstractTensor({', '.join(str(s) for s in self.size)}, dtype={self.dtype})"

    def create_intrinsic(self, ir_value: Value) -> Intrinsic:
        return IrValueTensor(ir_value, self.dtype)

    def get_ir_type(self, builder: ModuleBuilder) -> IrType:
        element_type = builder.torch_dtype_to_iree_type(self.dtype)
        with Location.unknown(builder.context):
            tensor_type = RankedTensorType.get(
                [s if s is not None else -1 for s in self.size], element_type
            )
        return tensor_type


def abstractify_single_value(value) -> AbstractTypedef:
    if isinstance(value, AbstractTypedef):
        return value
    if isinstance(value, Abstractifiable):
        return value.get_abstract_typedef()
    if isinstance(value, torch.Tensor):
        return AbstractTensor(*value.shape, dtype=value.dtype)
    raise TypeError(
        f"Cannot convert type {value.__class__} to an abstract type: {value}"
    )


def abstractify(tree):
    if isinstance(tree, TreeAbstractifiable):
        return tree.abstractify_tree()
    return tree_map(abstractify_single_value, tree)


###############################################################################
# Globals
###############################################################################


class LiveGlobalCollectionProxy:
    """Proxy object around a collection which knows how to redirect setitem."""

    __slots__ = ["_raw_collection"]

    def __init__(self, raw_collection):
        self._raw_collection = raw_collection

    def __getitem__(self, key: str):
        actual = self._raw_collection[key]
        if isinstance(actual, MaterializedGlobal):
            return actual
        else:
            return LiveGlobalCollectionProxy(actual)

    def __setitem__(self, key, value):
        item = self._raw_collection[key]
        if isinstance(item, MaterializedGlobal):
            current_ir_trace().handle_assignment(self, item, value)
        else:
            raise AttributeError(
                f"Globals collection {self._raw_collection.__class__} only supports assignment of leaves"
            )

    def __len__(self):
        return len(self._raw_collection)

    def __repr__(self):
        return f"LiveGlobalsProxy({self._raw_collection})"


class GlobalsDef:
    """Base class for all exporting descriptors."""

    __slots__ = [
        "_initialize",
        "_mutable",
    ]

    def __init__(self, *, initialize: bool, mutable: bool):
        self._initialize = initialize
        self._mutable = mutable

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """Yields tuples of name/value exports."""
        raise NotImplementedError

    def schema(self) -> TreeSpec:
        """A schema used to unflatten for access from Python."""
        raise NotImplementedError

    def track(self, module_builder: ModuleBuilder, export_namespace: str) -> Any:
        """Track the given pack of globals, returning a struct that can be used to access them."""
        flat_globals = []
        for name, value in self.items():
            # Switch on types we support.
            if isinstance(value, torch.Tensor):
                fq_name = f"{export_namespace}.{name}"
                mapping = module_builder.global_ref_tracker.track(value)
                if not mapping.is_empty:
                    logger.debug(
                        "IGNORE EXISTING TRACKED TENSOR(%s): %r", fq_name, mapping
                    )
                    flat_globals.append(mapping.value)
                    continue
                (
                    actual_symbol_name,
                    global_op,
                    global_type,
                ) = module_builder.create_tensor_global(
                    f"_{fq_name}",
                    value,
                    initialize=self._initialize,
                    mutable=self._mutable,
                )
                mapping.value = MaterializedGlobal(
                    fq_name,
                    self,
                    symbol_name=actual_symbol_name,
                    global_op=global_op,
                    global_type=global_type,
                )
                logger.debug("TRACK NEW TENSOR(%s): %r", fq_name, mapping)
                flat_globals.append(mapping.value)
                continue

            raise TypeError(f"Unsupported global type: {value.__class__}")
        tree_globals = tree_unflatten(flat_globals, self.schema())
        if isinstance(tree_globals, MaterializedGlobal):
            return tree_globals
        else:
            return LiveGlobalCollectionProxy(tree_globals)


class MaterializedGlobal(Intrinsic):
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

    def resolve_ir_values(self, trace: IrTrace) -> Sequence[Value]:
        with trace.loc, trace.ip:
            value = util_d.GlobalLoadOp(self.global_type, self.symbol_name).result
        return [value]

    def resolve_assignment(self, proc_trace: "IrTrace", ir_values: Sequence[Value]):
        if len(ir_values) != 1:
            raise ValueError(
                f"Can only assign a single value to a global. Got {len(ir_values)}"
            )
        source_ir_type = ir_values[0].type
        if source_ir_type != self.global_type:
            raise TypeError(
                f"Cannot assign to a global with a different type: {self.global_type} != {source_ir_type}"
            )
        with proc_trace.loc, proc_trace.ip:
            util_d.GlobalStoreOp(ir_values[0], self.symbol_name)

    def __repr__(self):
        return f"<MaterializedGlobal {self.export_name} = {self.symbol_name}:{self.global_type}>"

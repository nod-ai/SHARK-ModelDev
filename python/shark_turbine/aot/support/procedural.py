# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

from contextlib import contextmanager

import torch

from .ir_imports import (
    IrType,
    Location,
    Operation,
    RankedTensorType,
    StringAttr,
    Value,
    func_d,
    util_d,
)

from .ir_utils import (
    FunctionBuilder,
    ModuleBuilder,
)

from .utils import (
    TreeSpec,
    logger,
    thread_state,
    tree_flatten,
    tree_map,
    tree_unflatten,
    treespec_dumps,
)

###############################################################################
# Tracing intrinsics
###############################################################################


class ProcedureTraceError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


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
        trace_scopes = thread_state.trace_scopes
    except AttributeError:
        trace_scopes = thread_state.trace_scopes = []
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


###############################################################################
# Concrete procedure building IrTracer.
###############################################################################


class ProcedureTrace(IrTrace):
    """Captures execution of a Python func into IR."""

    __slots__ = [
        "proxy_posargs",
        "proxy_kwargs",
    ]

    def __init__(
        self,
        *,
        module_builder: ModuleBuilder,
        func_op: func_d.FuncOp,
        proxy_posargs,
        proxy_kwargs,
    ):
        super().__init__(module_builder=module_builder, func_op=func_op)
        self.proxy_posargs = proxy_posargs
        self.proxy_kwargs = proxy_kwargs

    @staticmethod
    def define_func(
        module_builder: ModuleBuilder,
        *,
        symbol_name: str,
        posargs: Sequence,
        kwargs: dict,
        loc: Location,
    ) -> "ProcedureTrace":
        # Unpack arguments.
        arguments_flat, arguments_tree_def = tree_flatten((posargs, kwargs))
        argument_ir_types = []
        for arg in arguments_flat:
            if not isinstance(arg, AbstractIntrinsic):
                raise ProcedureTraceError(f"Expected a AbstractIntrinsic but got {arg}")
            argument_ir_types.append(arg.get_ir_type(module_builder))

        with loc:
            _, func_op = module_builder.create_func_op(symbol_name, argument_ir_types)

        # Bind proxy arguments to an IR value.
        ir_proxy_arguments_flat = []
        for ir_value, arg_proxy_type in zip(
            func_op.body.blocks[0].arguments, arguments_flat
        ):
            ir_proxy_arguments_flat.append(arg_proxy_type.create_intrinsic(ir_value))

        # Unflatten.
        proxy_posargs, proxy_kwargs = tree_unflatten(
            ir_proxy_arguments_flat, arguments_tree_def
        )

        # Metadata.
        if arguments_flat:
            func_op.attributes["torch.args_schema"] = StringAttr.get(
                treespec_dumps(arguments_tree_def), context=module_builder.context
            )

        return ProcedureTrace(
            module_builder=module_builder,
            func_op=func_op,
            proxy_posargs=proxy_posargs,
            proxy_kwargs=proxy_kwargs,
        )

    def trace_py_func(self, py_f: Callable):
        with new_ir_trace_scope(self) as t:
            # TODO: Create IR proxies for python arguments.
            return_py_value = _unproxy(py_f(*self.proxy_posargs, **self.proxy_kwargs))
            if return_py_value is None:
                self.emit_return()
            else:
                flat_return_py_values, schema = tree_flatten(return_py_value)
                flat_return_ir_values = []
                for py_value in flat_return_py_values:
                    flat_return_ir_values.extend(convert_py_value_to_ir(self, py_value))
                self.func_op.attributes["torch.return_schema"] = StringAttr.get(
                    treespec_dumps(schema), context=self.context
                )
                self.emit_return(*flat_return_ir_values)

    def handle_call(self, target: Intrinsic, args, kwargs):
        """Implements calls to jittable functions."""
        with self.loc, self.ip:
            return target.resolve_call(self, *args, **kwargs)

    def handle_assignment(self, scope, target, updated_value):
        logger.debug(
            "ASSIGN %r.%r = %r", scope.__class__, target.__class__, updated_value
        )
        self._recursive_assign(target, updated_value, set())

    def _recursive_assign(self, target, source, encountered_set):
        target = _unproxy(target)
        source = _unproxy(source)

        # Check for cycles.
        target_id = id(target)
        if target_id in encountered_set:
            raise TypeError(f"Cycle in tree assignment target")
        encountered_set.add(target_id)

        # Leaves/terminals.
        if isinstance(target, Intrinsic):
            if not isinstance(source, Intrinsic):
                raise TypeError(
                    f"Cannot assign mismatched leaf types in a tree: "
                    f"{target.__class__} vs {source.__class__}"
                )
            leaf_values = source.resolve_ir_values(self)
            target.resolve_assignment(self, leaf_values)
            return

        # Zip across dicts.
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(
                    f"Mismatched dict assignment in a tree: {target.__class__} vs {source.__class__}"
                )
            target_keys = target.keys()
            source_keys = source.keys()
            if target_keys != source_keys:
                raise TypeError(
                    f"Mismatched dict keys in tree assignment: {target_keys} vs {source_keys}"
                )
            for k in target_keys:
                target_child = target[k]
                source_child = source[k]
                self._recursive_assign(target_child, source_child, encountered_set)
            return

        # Zip across lists/tuples (we let them be used interchangeably at the source).
        if isinstance(target, list):
            if not isinstance(source, (list, tuple)):
                if len(target) != len(source):
                    raise TypeError(
                        f"Mismatched sequence length in tree assignment: {len(target)} vs {len(source)}"
                    )
            for target_child, source_child in zip(target, source):
                self._recursive_assign(target_child, source_child, encountered_set)
            return

        raise TypeError(
            f"Cannot recursively assign through a container of {target.__class__}"
        )


def convert_py_value_to_ir(
    proc_trace: ProcedureTrace, py_value: Any
) -> Sequence[Value]:
    """Given procedurally traced python values, type check and convert to IR."""
    if isinstance(py_value, Intrinsic):
        return py_value.resolve_ir_values(proc_trace)

    raise TypeError(
        f"Illegal type passed in procedural trace: {py_value.__class__} ({py_value})"
    )


def _unproxy(value):
    if isinstance(value, LiveGlobalCollectionProxy):
        return value._raw_collection
    return value

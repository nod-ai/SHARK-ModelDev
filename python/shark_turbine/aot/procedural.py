# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
import re
import threading
from typing import Any, Callable, List, Sequence

import torch

from iree.compiler.ir import (
    FunctionType,
    InsertionPoint,
    Location,
    RankedTensorType,
    Type as IrType,
    TypeAttr,
    Value,
)

from iree.compiler.dialects import (
    func as func_d,
)

from torch.utils._pytree import (
    tree_map,
    tree_flatten,
    tree_unflatten,
)

from numpy import number
from collections.abc import Mapping

from .builder import ModuleBuilder
from ..dynamo.importer import TORCH_DTYPE_TO_MLIR_TYPE_ASM

# We need the inverse of the TORCH_DTYPE_TO_MLIR_TYPE_ASM table.
MLIR_TYPE_ASM_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_MLIR_TYPE_ASM.items()}

_thread_state = threading.local()


class Intrinsic:
    """Objects which interact natively with the tracing system implement this."""

    __slots__ = []

    def resolve_ir_values(self, proc_trace: "ProcedureTrace") -> Sequence[Value]:
        raise NotImplementedError(
            f"Cannot use {self} as an expression in a procedural function"
        )

    def resolve_call(self, proc_trace: "ProcedureTrace", *args, **kwargs):
        raise NotImplementedError(
            f"Cannot use {self} as the target of a call in a procedural function"
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


class IrTrace:
    """Gets callbacks for tracing events."""

    __slots__ = []

    def finalize(self):
        """Called when the trace is finished (popped off the stack)."""
        pass

    def handle_call(self, target: Intrinsic, args, kwargs):
        raise NotImplementedError(f"The current trace scope does not support calls")


class ImmediateIrTrace(IrTrace):
    __slots__ = []
    ...


def _trace_scopes() -> List[IrTrace]:
    try:
        trace_scopes = _thread_state.trace_scopes
    except AttributeError:
        trace_scopes = _thread_state.trace_scopes = [ImmediateIrTrace()]
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


class ProcedureTrace(IrTrace):
    """Captures execution of a Python func into IR."""

    __slots__ = [
        "module_builder",
        "func_op",
        "context",
        "ip",
        "return_types",
        "loc",
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
        self.module_builder = module_builder
        self.func_op = func_op
        self.context = func_op.context
        self.ip = InsertionPoint(self.func_op.entry_block)
        self.return_types = None
        self.loc = self.func_op.location
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

        return ProcedureTrace(
            module_builder=module_builder,
            func_op=func_op,
            proxy_posargs=proxy_posargs,
            proxy_kwargs=proxy_kwargs,
        )

    def trace_py_func(self, py_f: Callable):
        with new_ir_trace_scope(self) as t:
            # TODO: Create IR proxies for python arguments.
            return_py_value = py_f(*self.proxy_posargs, **self.proxy_kwargs)
            if return_py_value is None:
                self.emit_return()
            else:
                flat_return_py_values, _ = tree_flatten(return_py_value)
                flat_return_ir_values = []
                for py_value in flat_return_py_values:
                    flat_return_ir_values.extend(convert_py_value_to_ir(self, py_value))
                self.emit_return(*flat_return_ir_values)

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

    def handle_call(self, target: Intrinsic, args, kwargs):
        """Implements calls to jittable functions."""
        with self.loc, self.ip:
            return target.resolve_call(self, *args, **kwargs)


class ProcedureTraceError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def convert_py_value_to_ir(
    proc_trace: ProcedureTrace, py_value: Any
) -> Sequence[Value]:
    """Given procedurally traced python values, type check and convert to IR."""
    if isinstance(py_value, Intrinsic):
        return py_value.resolve_ir_values(proc_trace)

    raise TypeError(
        f"Illegal type passed in procedural trace: {py_value.__class__} ({py_value})"
    )


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
        return torch.empty(shape, dtype=self.dtype, device="meta")


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

    def resolve_ir_values(self, proc_trace: ProcedureTrace) -> Sequence[Value]:
        return (self.ir_value,)

# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Base classes for registering custom operations with the PyTorch
dispatcher.
"""

from typing import Any, Callable, Optional, Type, Union

from abc import ABC, abstractmethod, abstractproperty
import functools
import logging

import torch
from torch import Tensor

from ...support.ir_imports import (
    Block,
    Context,
    FunctionType,
    InsertionPoint,
    Location,
    StringAttr,
    SymbolTable,
    IrType,
    Value,
    builtin_d,
    func_d,
)

from ...support.conversions import (
    TORCH_DTYPE_TO_IREE_TYPE_ASM,
)

__all__ = [
    "ArgDescriptor",
    "CustomOp",
    "FreeFuncKernelBuilder",
    "IntArg",
    "KernelBuilder",
    "KernelSelection",
    "TensorArg",
]

logger = logging.getLogger("turbine.runtime.op_reg")

###############################################################################
# Op library management
###############################################################################

# All such custom kernels are registered in the 'turbine' library/namespace.
# We also allow extending existing libraries outside of this, but that is
# the non default case.
TURBINE_LIBRARY = torch.library.Library("turbine", "DEF")


class CustomOp(ABC):
    """Users subclass this in order to register a turbine custom op."""

    @staticmethod
    def register(
        op_class: Optional[Type["CustomOp"]],
        *,
        library: torch.library.Library = TURBINE_LIBRARY,
        dispatch_key: str = "",
        register_meta: bool = True,
        register_impl: bool = True,
    ) -> Callable:
        """Class decorator for `CustomOp` implementations.

        The decorator will instantiate the class and then replace it with
        the callable operation that can be used to invoke the kernel.

        Typical usage:

        ```
        @CustomOp.register
        class identity(CustomOp):
          ...

        result = identity(torch.tensor(1, 2, 3))
        ```
        """
        if not op_class:
            return functools.partial(
                CustomOp.register,
                library=library,
                dispatch_key=dispatch_key,
                register_meta=register_meta,
                register_impl=register_impl,
            )
        instance = op_class(
            library=library,
            dispatch_key=dispatch_key,
            register_meta=register_meta,
            register_impl=register_impl,
        )
        return instance.op

    def __init__(
        self,
        *,
        library: torch.library.Library,
        dispatch_key: str,
        register_meta: bool,
        register_impl: bool,
    ):
        name = self.name
        fq_schema = f"{name}{self.signature}"
        library.define(fq_schema)
        self.library = library
        self.cache_key_base = f"{library.ns}.{library.kind}::{name}"
        self.op = _get_library_op(library, name)

        # The meta kernel can be provided by the selection machinery and
        # does not require a tie-in to the kernel generator, which layers
        # on top.
        if register_meta:
            library.impl(name, _get_meta_impl(self), "Meta")

        if register_impl:
            library.impl(name, _create_impl_trampoline(self), dispatch_key)

    @abstractproperty
    def name(self) -> str:
        """Name of the operation."""
        ...

    @abstractproperty
    def signature(self) -> str:
        """PyTorch function signature.

        This excludes the name, which will come from the `name` property
        and be prepended to make a full PyTorch schema.
        """
        ...

    @abstractmethod
    def select(self, sel: "KernelSelection"):
        """Performs kernel selection.

        This method has three purposes:

          1. Selects which kernel specialization is needed based on
             arguments.
          2. Returns the meta tensor results of the operation, effectively
             completing the transfer function from argument types to
             result types.
          3. Sets additional metadata that the generate method can use.

        The `device="meta"` kernel implementation is composed completely by
        invoking `select`. For implementation devices, `select` is called
        for each invocation. The `generate` will be called subsequently if
        the kernel needs to be generated.
        """
        ...

    @abstractmethod
    def generate(self, ksel: "KernelSelection", kb: "KernelBuilder"):
        """Generates a kernel based on the `KernelSelection`.

        This method should generate IR into the given `KernelBuilder`. It
        can do so by consulting any state set on the `KernelSelection`.
        Each `KernelSelection.args` corresponds to `KernelBuilder.args`.
        Unless if the argument was set as `is_ir_arg=False`, the argument
        will be a `Value`. Otherwise, it will be `None`. It is recommended
        to use `KernelBuilder.arg(n)` to access.

        Generation should conclude with a call to `KernelBuilder.yield_results`.
        """
        ...


class KernelSelection:
    """Represents a selected kernel based on a concrete signature.

    The `CustomOp.select` method must yield an instance of this, and
    it will be done for every invocation. At this point, the kernel
    has not yet been generated, but we have selected a generation
    strategy based on a concrete signature.

    This mechanism also serves as the means for servicing `meta`
    registrations because it implicitly computes everything needed
    (i.e. shapes, etc).
    """

    __slots__ = [
        "args",
        "arg_descs",
        "op",
        "result_descs",
        "variant",
    ]

    def __init__(self, op: CustomOp, args: list[Any]):
        self.op = op
        self.args = args
        self.arg_descs: list[Optional[ArgDescriptor]] = len(args) * [None]
        self.result_descs: list[ArgDescriptor] = []
        self.variant: str = "default"

    def generate_meta_returns(self) -> Any:
        results = [d.generate_meta() for d in self.result_descs]
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    @property
    def spec_key(self) -> str:
        arg_keys = ",".join(d.spec_key for d in self.arg_descs)
        return_keys = ",".join(d.spec_key for d in self.result_descs)
        return f"{self.op.cache_key_base}::{self.variant}({arg_keys})->({return_keys})"

    def arg_tensor(self, arg: int) -> "TensorArg":
        """Declares an argument to allow any ranked tensor and to specialize for each rank
        and dtype.

        Returns the argument descriptor, which can be used to further inspect or constrain
        the selection. It will default to allowing all dimensions to be dynamic.
        """
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, Tensor
        ), f"Argument type mismatch from Torch for {arg}: Expected tensor, got {type(arg_value)}"
        arg_descs[arg] = desc = TensorArg(arg_value)
        return desc

    def arg_int(self, arg: int) -> "IntArg":
        """Declares an argument to be an integer value that can take any value.

        Returns the argument descriptor, which can be used to further inspect or constrain
        the selection.
        """
        arg_descs = self.arg_descs
        arg_value = self.args[arg]
        assert arg_descs[arg] is None, f"Already constrained argument {arg}"
        assert isinstance(
            arg_value, int
        ), f"Argument type mismatch from Torch for {arg}: Expected int, got {type(arg_value)}"
        arg_descs[arg] = desc = IntArg(arg_value)
        return desc

    def return_tensor(self, t: Tensor) -> "TensorArg":
        """Marks the next return value as a Tensor.

        By default, it will be rank and dtype specialized but have completely dynamic
        dimensions. Dimensions can be further constrained by modifying the returned
        descriptor.
        """
        desc = TensorArg(t)
        self.result_descs.append(desc)
        return desc


class TensorArg:
    __slots__ = [
        "t",
        "spec_dims",
        "is_ir_arg",
        "maybe_tensor_value",
    ]

    def __init__(self, t: Tensor):
        self.t = t
        # Any static dims that we are specializing. Defaults to all dynamic.
        self.spec_dims: list[Optional[int]] = len(t.shape) * [None]
        self.is_ir_arg = True
        # All descriptors have an attribute to indicate their value
        # as a tensor, and those that aren't are fixated to None.
        # This is to enable fast lookup in the hot path of determining
        # how to dispatch.
        self.maybe_tensor_value: Tensor = t

    def generate_meta(self) -> Tensor:
        t = self.t
        if t.device == "meta":
            return t
        else:
            return t.clone().detach().to("meta")

    @property
    def spec_key(self) -> str:
        """Generates a key that will be the same for all specializations."""
        t = self.t
        return f"tensor[{len(t.shape)}:{str(t.dtype)}]<{self.spec_dims}>"

    @property
    def mlir_type_asm(self) -> str:
        t = self.t
        try:
            dtype_asm = TORCH_DTYPE_TO_IREE_TYPE_ASM[t.dtype]
        except KeyError as e:
            raise KeyError(
                f"Unknown mapping of torch dtype {t.dtype} to MLIR "
                f"(possibly missing in TORCH_DTYPE_TO_IREE_TYPE_ASM table)"
            ) from e
        dim_asm = "x".join(["?" if d is None else str(d) for d in self.spec_dims])
        spec = f"{dim_asm}x{dtype_asm}" if dim_asm else dtype_asm
        return f"tensor<{spec}>"


class IntArg:
    __slots__ = [
        "is_ir_arg",
        "v",
        "spec_value",
        "maybe_tensor_value",
    ]

    def __init__(self, v: int):
        self.v = v
        self.spec_value: Optional[Any] = None
        self.is_ir_arg = True
        # All descriptors have an attribute to indicate their value
        # as a tensor, and those that aren't are fixated to None.
        # This is to enable fast lookup in the hot path of determining
        # how to dispatch.
        self.maybe_tensor_value: Optional[Tensor] = None

    def generate_meta(self) -> int:
        return self.v

    @property
    def spec_key(self) -> str:
        """Generates a key that will be the same for all specializations."""
        return f"int<{self.spec_value}>"

    @property
    def mlir_type_asm(self) -> str:
        # TODO: We can have individual kernels constrain this to a narrower
        # type.
        return "i64"


ArgDescriptor = Union[TensorArg, IntArg]

###############################################################################
# KernelBuilder
# Helper object for constructing IR
###############################################################################


class KernelBuilder(ABC):
    """Support class for building a kernel."""

    def __init__(
        self,
        ksel: KernelSelection,
        arg_bindings: list[Value],
        *,
        ip: InsertionPoint,
        module_body: Block,
        symbol_table: SymbolTable,
    ):
        self.ksel = ksel
        self.arg_bindings = arg_bindings
        self.ip = ip
        self.module_body = module_body
        self.symbol_table = symbol_table

    def arg_value(self, index: int) -> Value:
        """Gets the concrete IR `Value` for the argument at `index`.

        This will assert if the corresponding argument was set as `is_ir_arg=False`
        during kernel selection.
        """
        try:
            v = self.arg_bindings[index]
        except IndexError as e:
            raise AssertionError(
                f"Out of range access to kernel arg. Expected 0..{len(self.arg_bindings)}. Got {index}"
            ) from e
        assert (
            v is not None
        ), f"No `Value` is available for arg {index}: it was marked as `is_ir_arg=False` during kernel selection."
        return v

    @abstractmethod
    def yield_results(self, *results: Value):
        """Yields results of the kernel computation."""
        ...


class FreeFuncKernelBuilder(KernelBuilder):
    """Kernel builder that emits the body of the kernel into a free function.

    This is intended to be used when compiling a standalone module that will
    be directly invoked by the runtime. Further variants exist that generate
    into a func but also emit a call into another local context.
    """

    def __init__(
        self,
        ksel: KernelSelection,
        *,
        module_body: Block,
        symbol_table: SymbolTable,
        func_name: Optional[str] = None,
        is_public: bool = True,
    ):
        self.module_op = module_body.owner
        context = self.module_op.context
        if func_name is None:
            func_name = ksel.op.name
        with context, Location.unknown(), InsertionPoint(module_body):
            arg_types = [
                IrType.parse(d.mlir_type_asm) for d in ksel.arg_descs if d.is_ir_arg
            ]
            result_types = [IrType.parse(d.mlir_type_asm) for d in ksel.result_descs]
            ftype = FunctionType.get(arg_types, result_types)
            func_op = func_d.FuncOp(func_name, ftype)
            if not is_public:
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
            entry_block: Block = func_op.add_entry_block()
            symbol_table.insert(func_op)

        # Map inputs to arg bindings, lining up with arguments that are elided.
        block_arguments = list(entry_block.arguments)
        block_arguments.reverse()
        arg_bindings: list[Optional[Value]] = []
        for desc in ksel.arg_descs:
            if desc.is_ir_arg:
                arg_bindings.append(block_arguments.pop())
            else:
                arg_bindings.append(None)

        super().__init__(
            ksel,
            arg_bindings,
            ip=InsertionPoint(entry_block),
            module_body=module_body,
            symbol_table=symbol_table,
        )

    @staticmethod
    def create_module(
        ksel: KernelSelection,
        *,
        context: Optional[Context] = None,
        func_name: Optional[str] = None,
        is_public: bool = True,
    ) -> "FreeFuncKernelBuilder":
        """Short-cut to create a new module with a single function in one shot."""
        if context is None:
            context = Context()
        with context, Location.unknown():
            module_op = builtin_d.ModuleOp()
            return FreeFuncKernelBuilder(
                ksel,
                module_body=module_op.body,
                symbol_table=SymbolTable(module_op),
                func_name=func_name,
                is_public=is_public,
            )

    def yield_results(self, *results: Value):
        """Yields results of the kernel computation."""
        with self.ip, Location.unknown():
            func_d.ReturnOp(results)


###############################################################################
# Private utilities
###############################################################################


def _get_library_op(library: torch.library.Library, name: str) -> Any:
    ns = getattr(torch.ops, library.ns)
    return getattr(ns, name)


def _get_meta_impl(op: CustomOp):
    def meta(*args):
        sel = KernelSelection(op, args)
        op.select(sel)
        if logger.isEnabledFor(logging.DEBUG):
            logging.debug(
                "Meta dispatch on %s for specialization %s", op.name, sel.spec_key
            )
        return sel.generate_meta_returns()

    return meta


def _create_impl_trampoline(op: CustomOp):
    # Import lazily when an implementation trampoline is requested to avoid
    # circular dependency between base objects and eager runtime goo.
    from .eager import (
        eager_dispatch,
    )

    def handler(*args):
        ksel = KernelSelection(op, args)
        op.select(ksel)
        if logger.isEnabledFor(logging.DEBUG):
            logging.debug(
                "Dispatch on %s for specialization %s", op.name, ksel.spec_key
            )
        return eager_dispatch(ksel)

    return handler

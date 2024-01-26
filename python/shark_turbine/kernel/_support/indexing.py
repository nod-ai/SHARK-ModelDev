from typing import Any, ClassVar, Optional, Type, TypeVar, Union, cast

from abc import ABC, abstractmethod
from enum import Enum

import torch

from .. import ops

from . import context

__all__ = [
    "KernelBuffer",
    "Grid",
    "InputBuffer",
    "OutputBuffer",
    "IndexExpr",
    "IndexSymbol",
    "TemporaryBuffer",
    "sym",
    "sym_0",
    "sym_1",
    "sym_2",
    "sym_n1",
]


class NotSetType:
    ...


NotSet = NotSetType()

SubtypeT = TypeVar("SubtypeT")

###############################################################################
# ElementType
###############################################################################


class ElementType(ABC):
    @staticmethod
    def cast(something) -> "ElementType":
        if isinstance(something, torch.dtype):
            return TorchElementType(something)
        else:
            raise TypeError(
                f"Cannot convert {something} (of type {type(something)}) to an element type"
            )

    @abstractmethod
    def ir_type_asm(self) -> str:
        ...


class TorchElementType(ElementType):
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __repr__(self):
        return repr(self.dtype)

    def __eq__(self, other):
        return isinstance(other, TorchElementType) and self.dtype == other.dtype

    def ir_type_asm(self) -> str:
        dtype = self.dtype
        if dtype == torch.float32:
            return "f32"
        else:
            raise ValueError(f"Torch dtype {dtype} cannot be mapped to MLIR type")


DefaultElementType = TorchElementType(torch.float32)

###############################################################################
# Dimension symbols
###############################################################################

import sympy

IndexExpr = sympy.core.Expr
IndexSymbol = sympy.core.Symbol


def index_symbol(name: str) -> IndexSymbol:
    symbol = sympy.symbols(name)
    if not isinstance(symbol, sympy.core.Symbol):
        raise ValueError(f"Expected a single symbol name but got '{name}'")
    return symbol


class _IndexSymbolExpando:
    def __getattr__(self, n):
        return index_symbol(n)

sym = _IndexSymbolExpando()

sym_0 = index_symbol("0")
sym_1 = index_symbol("1")
sym_2 = index_symbol("2")
sym_n1 = index_symbol("-1")


###############################################################################
# Grid
###############################################################################


class _GridMeta(type):
    """Meta-class for a symbolically shaped grid."""

    def __new__(
        mcls,
        name: str,
        bases,
        dct,
        *,
        symbolic_shape: Optional[tuple[IndexExpr]],
    ):
        new_class = type.__new__(mcls, name, bases, dct)
        new_class.symbolic_shape = symbolic_shape
        new_class.rank = len(symbolic_shape) if symbolic_shape is not None else None
        new_class.__qualname__ = repr(new_class)
        return new_class

    def __class_getitem__(
        cls, symbolic_shape: Union[IndexExpr, tuple[IndexExpr]]
    ) -> Type["Grid"]:
        if not isinstance(symbolic_shape, tuple):
            symbolic_shape = (symbolic_shape,)
        return cast(Grid, _make_shaped_grid(cls, symbolic_shape))

    def __repr__(self):
        if self.symbolic_shape:
            return f"Grid[{', '.join(s.name for s in self.symbolic_shape)}]"
        else:
            return "Grid"


class Grid(metaclass=_GridMeta, symbolic_shape=None):
    """Grid with bounding symbolic shape information in the type."""

    symbolic_shape: ClassVar[Optional[tuple[IndexExpr]]]
    rank: int

    def __init__(self, *dims: int):
        rank = len(dims)
        if self.symbolic_shape is not None:
            if rank != len(self.symbolic_shape):
                raise ValueError(
                    f"Cannot create {type(self)}({', '.join(str(i) for i in dims)}): mismatched symbolic rank"
                )

        self.dims = dims
        # Shadow the type rank with the actual, which makes it concrete
        # for the generic case.
        self.rank = rank

    def __repr__(self):
        return f"{repr(type(self))}({', '.join(str(i) for i in self.dims)})"

    def __getitem__(self, index: int) -> int:
        return self.dims[index]

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)


def _make_shaped_grid(cls: Type[Grid], symbolic_shape: tuple[IndexExpr]):
    class ShapedGrid(Grid, symbolic_shape=symbolic_shape):
        ...

    return ShapedGrid


###############################################################################
# KernelBuffer
###############################################################################


class KernelBufferUsage(Enum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2
    TEMPORARY = 3

    @staticmethod
    def _type_name(v) -> str:
        if v == KernelBufferUsage.NONE:
            return "KernelBuffer"
        elif v == KernelBufferUsage.INPUT:
            return "InputBuffer"
        elif v == KernelBufferUsage.OUTPUT:
            return "OutputBuffer"
        elif v == KernelBufferUsage.TEMPORARY:
            return "TemporaryBuffer"
        else:
            raise AssertionError(f"uncovered KernelBufferUsage enum ({v})")


class _KernelBufferMeta(type):
    """Meta-class for kernel buffers.

    This lets us specialize with symbolic shape information.
    """

    element_type: ElementType
    usage: KernelBufferUsage
    symbolic_shape: Optional[tuple[IndexExpr]]
    rank: Optional[int]

    def __new__(
        mcls,
        name: str,
        bases,
        dct,
    ):
        element_type = dct.get("element_type") or DefaultElementType
        dct["element_type"] = element_type
        usage = dct.get("usage") or KernelBufferUsage.NONE
        dct["usage"] = usage
        if "usage" not in dct:
            dct["usage"] = KernelBufferUsage.NONE
        symbolic_shape = dct.get("symbolic_shape")
        dct["symbolic_shape"] = symbolic_shape
        dct["rank"] = len(symbolic_shape) if symbolic_shape is not None else None
        dct["__qualname__"] = _kernel_buffer_type_repr(
            element_type=element_type, usage=usage, symbolic_shape=symbolic_shape
        )
        new_class = type.__new__(mcls, name, bases, dct)
        return new_class

    def new_subtype(
        cls: Type[SubtypeT],
        *,
        element_type: Union[NotSetType, ElementType] = NotSet,
        symbolic_shape: Union[NotSetType, Optional[tuple[IndexExpr]]] = NotSet,
        usage: Union[NotSetType, KernelBufferUsage] = NotSet,
    ) -> Type[SubtypeT]:
        init_element_type = (
            element_type if element_type is not NotSet else cls.element_type
        )
        init_symbolic_shape = (
            symbolic_shape if symbolic_shape is not NotSet else cls.symbolic_shape
        )
        init_usage = usage if usage is not NotSet else cls.usage

        class Subtype(cls):
            element_type = init_element_type
            symbolic_shape = init_symbolic_shape
            usage = init_usage

        return Subtype

    def of(
        cls: Type[SubtypeT], element_type: Union[Any, ElementType, torch.dtype]
    ) -> Type[SubtypeT]:
        return cls.new_subtype(element_type=element_type)

    def __repr__(cls):
        return _kernel_buffer_type_repr(
            element_type=cls.element_type,
            usage=cls.usage,
            symbolic_shape=cls.symbolic_shape,
        )


def is_kernel_buffer_meta_derived(t: type) -> bool:
    return isinstance(t, _KernelBufferMeta)


def _kernel_buffer_type_repr(
    *,
    element_type: ElementType,
    usage: KernelBufferUsage,
    symbolic_shape: Optional[tuple[IndexExpr]],
) -> str:
    root = KernelBufferUsage._type_name(usage)
    if symbolic_shape:
        stem = f"{root}[{', '.join(s.name for s in symbolic_shape)}]"
    else:
        stem = f"{root}"
    if element_type != DefaultElementType:
        stem += f".of({element_type})"
    return stem


class KernelBuffer(metaclass=_KernelBufferMeta):
    """Represents a buffer in global memory.

    Top level kernels always operate on global memory via these
    buffers, and the primary operations that can be performed on
    them are loads/stores and DMAs to some form of compute
    capable local buffer.

    When executing eagerly, these are backed by a normal torch
    Tensor. When compiling, an appropriate duck-typed proxy
    is used.
    """

    usage: ClassVar[KernelBufferUsage]
    symbolic_shape: ClassVar[Optional[tuple[IndexExpr]]]
    rank: Optional[int]

    def __init__(self, tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor but got {tensor}"
        type_rank = type(self).rank
        tensor_rank = len(tensor.shape)
        if type_rank is not None and type_rank != tensor_rank:
            raise ValueError(
                f"Cannot create {type(self)}(tensor({tensor.shape})): mismatched symbolic rank"
            )
        self._tensor = tensor
        self.rank = tensor_rank

    def __class_getitem__(
        cls, symbolic_shape: Union[IndexExpr, tuple[IndexExpr]]
    ) -> Type["KernelBuffer"]:
        if not isinstance(symbolic_shape, tuple):
            symbolic_shape = (symbolic_shape,)
        return cast(cls, cls.new_subtype(symbolic_shape=symbolic_shape))

    def __repr__(self):
        return f"{type(self)}({self._tensor})"

    def __setitem__(self, key, item):
        ops.kernel_buffer_setitem(self, key, item)

    def __getitem__(self, key):
        return ops.kernel_buffer_getitem(self, key)


class InputBuffer(KernelBuffer):
    usage = KernelBufferUsage.INPUT


class OutputBuffer(KernelBuffer):
    usage = KernelBufferUsage.OUTPUT


class TemporaryBuffer(KernelBuffer):
    usage = KernelBufferUsage.TEMPORARY


###############################################################################
# IndexingContext
###############################################################################


class IndexingContext:
    """The indexing context is responsible handling the binding of indexed
    symbols to concrete values.
    """

    __tk_context_idname__ = "IndexingContext"

    def __init__(self):
        self.constant_bindings: dict[IndexSymbol, int] = {
            sym_0: 0,
            sym_1: 1,
            sym_2: 2,
            sym_n1: -1,
        }

    def bind_constant(self, sym: IndexSymbol, value: int):
        existing = self.constant_bindings.get(sym)
        if existing is not None and existing != value:
            raise ValueError(
                f"Attempt to rebind symbol {sym} to different constant ({value} vs {existing})"
            )
        self.constant_bindings[sym] = value

    def get_static_value(self, sym: IndexExpr) -> Optional[int]:
        """If the symbol can be resolved to a static value, returns it."""
        return self.constant_bindings.get(sym)

    ##### Context management.
    @staticmethod
    def current() -> "IndexingContext":
        return context.current(IndexingContext)

    def __enter__(self) -> "IndexingContext":
        return context.push(IndexingContext, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pop(IndexingContext, self)

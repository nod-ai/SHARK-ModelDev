from typing import Any, ClassVar, Optional, Type, TypeVar, Union, cast

from abc import ABC, abstractmethod
from enum import Enum

import torch

from .. import ops

from . import context

__all__ = [
    "BoundedSymbolicValue",
    "KernelBuffer",
    "Grid",
    "InputBuffer",
    "OutputBuffer",
    "SymbolDef",
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


class SymbolExpr:
    def is_one(self) -> Optional[bool]:
        """Returns True if the symbol is known to be 1.

        Return False if known to be != 1 and None if not known.
        """
        raise NotImplementedError

    def is_non_negative(self) -> Optional[bool]:
        """Returns True is the symbol is known to be non-negative.

        Returns False if known to be negative and None if not known.
        """
        raise NotImplementedError

    def is_positive(self) -> Optional[bool]:
        """Returns True is the symbol is known to be greater than zero.

        Returns False if known to be <= 0 and None if not known.
        """
        raise NotImplementedError

    def is_negative(self) -> Optional[bool]:
        """Returns True is the symbol is known to be greater than zero.

        Returns False if known to be <= 0 and None if not known.
        """
        raise NotImplementedError


class SymbolDef(SymbolExpr):
    """Represents a named symbol representing a dimension in a shape."""

    ALL_SYMBOLS: ClassVar[dict[str, "SymbolDef"]] = dict()
    name: str

    def __new__(cls, name: str):
        existing = cls.ALL_SYMBOLS.get(name)
        if existing is not None:
            return existing
        new = super().__new__(cls)
        new.name = name
        cls.ALL_SYMBOLS[name] = new
        return new

    def __repr__(self):
        return f"Symbol({self.name})"

    @classmethod
    def create_expando(cls):
        """Create an expando class that creates unique symbols based on attr access."""

        class Expando:
            def __getattr__(self, n):
                return cls(n)

        return Expando()

    def is_one(self) -> Optional[bool]:
        value = IndexingContext.current().get_static_value(self)
        if value is None:
            return None
        return value == 1

    def is_non_negative(self) -> Optional[bool]:
        value = IndexingContext.current().get_static_value(self)
        if value is None:
            return None
        return value >= 0

    def is_positive(self) -> Optional[bool]:
        value = IndexingContext.current().get_static_value(self)
        if value is None:
            return None
        return value > 0

    def is_negative(self) -> Optional[bool]:
        value = IndexingContext.current().get_static_value(self)
        if value is None:
            return None
        return value < 0


sym = SymbolDef.create_expando()

sym_0 = SymbolDef("0")
sym_1 = SymbolDef("1")
sym_2 = SymbolDef("2")
sym_n1 = SymbolDef("-1")


###############################################################################
# Bounded symbolic value.
###############################################################################

BoundedRangeExprT = tuple[Optional[SymbolExpr], Optional[SymbolExpr]]


class _BoundedSymbolicValueMeta(type):
    """Meta-class for deriving new bounded symbolic values."""

    range: BoundedRangeExprT

    def __new__(mcls, name: str, bases, dct, *, range: BoundedRangeExprT):
        dct["range"] = range
        dct["__qualname__"] = _bounded_symbolic_value_repr(range=range)
        new_class = type.__new__(mcls, name, bases, dct)
        return new_class

    def __repr__(cls):
        return _bounded_symbolic_value_repr(range=cls.range)

    @property
    def min_bound(cls) -> Optional[SymbolExpr]:
        return cls.range[0]

    @property
    def max_bound(cls) -> Optional[SymbolExpr]:
        return cls.range[1]

    def bound(
        cls: Type[SubtypeT],
        min_bound: Optional[SymbolExpr],
        max_bound: Optional[SymbolExpr],
    ) -> Type[SubtypeT]:
        class Bounded(BoundedSymbolicValue, range=(min_bound, max_bound)):
            ...

        return Bounded

    def narrow(
        cls: Type[SubtypeT],
        *,
        min_bound: Optional[SymbolExpr] = None,
        max_bound: Optional[SymbolExpr] = None,
    ) -> Type[SubtypeT]:
        class Bounded(
            BoundedSymbolicValue,
            range=(
                min_bound if min_bound is not None else cls.min_bound,
                max_bound if max_bound is not None else cls.max_bound,
            ),
        ):
            ...

        return Bounded


def _bounded_symbolic_value_repr(*, range: BoundedRangeExprT) -> str:
    min_expr, max_expr = range
    min_s = repr(min_expr) if min_expr is not None else "*"
    max_s = repr(max_expr) if max_expr is not None else "*"
    return f"BoundedSymbolicValue({min_s} : {max_s})"


class BoundedSymbolicValue(
    SymbolExpr, metaclass=_BoundedSymbolicValueMeta, range=(None, None)
):
    """Represents a symbolic value that is bounded to a range fixed for the type."""

    def __init__(self, value: Optional[int] = None):
        self.value = value

    def __repr__(self):
        return f"{type(self)}({'proxy' if self.value is None else self.value})"

    @property
    def static_range(self) -> Optional[tuple[int, int]]:
        # TODO: This is a hack until shape derivation is in place.
        ctx = IndexingContext.current()
        mn, mx = type(self).range
        if mn is not None:
            mn = ctx.get_static_value(mn)
        if mx is not None:
            mx = ctx.get_static_value(mx)
        if mn is not None and mx is not None:
            return mn, mx
        else:
            return None

    def is_one(self) -> Optional[bool]:
        r = self.static_range
        if r:
            return r[0] == 1 and r[1] == 2
        return None

    def is_non_negative(self) -> Optional[bool]:
        r = self.static_range
        if r:
            return r[0] >= 0
        return None

    def is_positive(self) -> Optional[bool]:
        r = self.static_range
        if r:
            return r[0] > 0
        return None

    def is_negative(self) -> Optional[bool]:
        r = self.static_range
        if r:
            return r[1] < 0
        return None


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
        symbolic_shape: Optional[tuple[SymbolDef]],
    ):
        new_class = type.__new__(mcls, name, bases, dct)
        new_class.symbolic_shape = symbolic_shape
        new_class.rank = len(symbolic_shape) if symbolic_shape is not None else None
        new_class.__qualname__ = repr(new_class)
        return new_class

    def __class_getitem__(
        cls, symbolic_shape: Union[SymbolDef, tuple[SymbolDef]]
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

    symbolic_shape: ClassVar[Optional[tuple[SymbolDef]]]
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


def _make_shaped_grid(cls: Type[Grid], symbolic_shape: tuple[SymbolDef]):
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
    symbolic_shape: Optional[tuple[SymbolDef]]
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
        symbolic_shape: Union[NotSetType, Optional[tuple[SymbolDef]]] = NotSet,
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
    symbolic_shape: Optional[tuple[SymbolDef]],
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
    symbolic_shape: ClassVar[Optional[tuple[SymbolDef]]]
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
        cls, symbolic_shape: Union[SymbolDef, tuple[SymbolDef]]
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
        self.constant_bindings: dict[SymbolDef, int] = {
            sym_0: 0,
            sym_1: 1,
            sym_2: 2,
            sym_n1: -1,
        }

    def bind_constant(self, sym: SymbolDef, value: int):
        existing = self.constant_bindings.get(sym)
        if existing is not None and existing != value:
            raise ValueError(
                f"Attempt to rebind symbol {sym} to different constant ({value} vs {existing})"
            )
        self.constant_bindings[sym] = value

    def get_static_value(self, sym: SymbolExpr) -> Optional[int]:
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

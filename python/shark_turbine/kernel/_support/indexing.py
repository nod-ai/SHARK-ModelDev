from typing import ClassVar, Optional, Type, TypeVar, Union, cast
import torch

__all__ = [
    "KernelBuffer",
    "Grid",
    "SymbolDef",
    "sym",
]

Grid = tuple[int, ...]


###############################################################################
# Dimension symbols
###############################################################################


class SymbolDef:
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


sym = SymbolDef.create_expando()

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


class _KernelBufferMeta(type):
    """Meta-class for kernel buffers.

    This lets us specialize with symbolic shape information.
    """

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
    ) -> Type["KernelBuffer"]:
        if not isinstance(symbolic_shape, tuple):
            symbolic_shape = (symbolic_shape,)
        return cast(KernelBuffer, _make_shaped_kernel_buffer(cls, symbolic_shape))

    def __repr__(self):
        if self.symbolic_shape:
            return f"KernelBuffer[{', '.join(s.name for s in self.symbolic_shape)}]"
        else:
            return "KernelBuffer"


class KernelBuffer(metaclass=_KernelBufferMeta, symbolic_shape=None):
    """Represents a buffer in global memory.

    Top level kernels always operate on global memory via these
    buffers, and the primary operations that can be performed on
    them are loads/stores and DMAs to some form of compute
    capable local buffer.

    When executing eagerly, these are backed by a normal torch
    Tensor. When compiling, an appropriate duck-typed proxy
    is used.
    """

    symbolic_shape: ClassVar[Optional[tuple[SymbolDef]]]
    rank: int

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

    def __repr__(self):
        return f"{type(self)}({self._tensor})"

    def __setitem__(self, key, item):
        self._tensor.__setitem__(key, item)

    def __getitem__(self, key):
        return self._tensor.__getitem__(key)


def _make_shaped_kernel_buffer(
    cls: Type[KernelBuffer], symbolic_shape: tuple[SymbolDef]
):
    class ShapedKernelBuffer(KernelBuffer, symbolic_shape=symbolic_shape):
        ...

    return ShapedKernelBuffer

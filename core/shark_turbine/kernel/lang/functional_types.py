from typing import Type, TypeVar, cast, ClassVar
from enum import Enum
from .._support.indexing import IndexExpr
from .._support.shaped_type import ShapedDataType
from .._support.dtype import DataType, f32
import torch

__all__ = [
    "Memory",
    "Register",
    "AddressSpace",
]

MemoryTypeT = TypeVar("MemoryTypeT")

class AddressSpace(Enum):
    REGISTER = 0
    SHARED_MEMORY = 1
    GLOBAL_MEMORY = 2

class _MemoryStorage(ShapedDataType):
    def new_subtype(cls: Type[MemoryTypeT],
                    *,
                    symbolic_shape: tuple[IndexExpr, ...],
                    address_space: AddressSpace,
                    dtype: DataType) -> Type[MemoryTypeT]:
        init_symbolic_shape = symbolic_shape
        init_dtype = dtype
        init_address_space = address_space if address_space else AddressSpace.REGISTER.value

        class MemoryType(cls):
            symbolic_shape = init_symbolic_shape
            rank = len(symbolic_shape)
            address_space = init_address_space
            dtype = init_dtype

        return cast(Type[MemoryTypeT], MemoryType)


class Memory(metaclass=_MemoryStorage):
    """
    Represents storage anywhere in the memory hierarchy except registers.
    Parameterized by a shape, address space and element type. The allocated
    memory is traversed by an iterator that specifies the offset, stride
    and size along each dimension.
    """

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    address_space: ClassVar[int]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]

    def __init__(self, tensor: torch.Tensor) -> None:
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor but got {tensor}"
        self._tensor = tensor

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Memory"]:
        """Syntax: `Memory[shape1, ...., shapeN, addressSpace, dtype]"""
        if not isinstance(shape_and_dtype, tuple) or len(shape_and_dtype) < 3:
            raise TypeError(f"Expected at least 3 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-2]
        addressSpace = shape_and_dtype[-2]
        dtype = shape_and_dtype[-1]

        if not all(isinstance(s, IndexExpr) for s in shape):
            raise TypeError(f"Expected shape to be a tuple of IndexExpr, got {shape}")
        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")
        if not isinstance(addressSpace, IndexExpr):
            raise TypeError(f"Expected addressSpace to be a AddressSpace, got {addressSpace}")

        shape = cast(tuple[IndexExpr, ...], shape)
        dtype = cast(DataType, dtype)
        addressSpace = cast(AddressSpace, addressSpace)
        return cls.new_subtype(symbolic_shape=shape, address_space=addressSpace, dtype=dtype)

class Register(metaclass=_MemoryStorage):
    "Represents virtual registers. Parameterized by a shape and element type."
    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]

    def __init__(self, shape, dtype) -> None:
        if not isinstance(shape, tuple):
            raise TypeError(f"Expected at shape to be a tuple, got: {shape}")
        self.symbolic_shape = shape
        self.rank = len(self.symbolic_shape)
        self.dtype = dtype
        self.value = None

    def set(self, value) -> None:
        self.value = value

    def __class_getitem__(cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
                      ) -> Type["Register"]:

        if not isinstance(shape_and_dtype, tuple) or len(shape_and_dtype) < 2:
            raise TypeError(f"Expected at least 2 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        dtype = shape_and_dtype[-1]

        shape = cast(tuple[IndexExpr, ...], shape)
        dtype = cast(DataType, dtype)
        return cls.new_subtype(symbolic_shape=shape, dtype=dtype, address_space=AddressSpace.REGISTER.value)

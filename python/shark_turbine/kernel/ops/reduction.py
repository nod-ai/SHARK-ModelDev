from typing import Any, List
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "vector_max",
    "vector_sum",
]


@define_op
def vector_max(source: "Vector", dims: List[int]) -> "Vector":
    ...


@define_op
def vector_sum(source: "Vector", dims: List[int]) -> "Vector":
    ...

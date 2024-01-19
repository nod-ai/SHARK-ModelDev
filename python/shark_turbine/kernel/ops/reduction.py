from typing import Any, List
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "vector_dot",
]


@define_op
def vector_dot(lhs: "Vector", rhs: "Vector", acc) -> "Vector":
    ...

from typing import Any, Tuple
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "vector_add",
    "vector_sub",
    "vector_mul",
    "vector_div",
    "vector_exp",
    "vector_zeros"
]


@define_op
def vector_add(lhs: "Vector", rhs: "Vector") -> "Vector":
    ...


@define_op
def vector_sub(lhs: "Vector", rhs: "Vector") -> "Vector":
    ...


@define_op
def vector_mul(lhs: "Vector", rhs: "Vector") -> "Vector":
    ...


@define_op
def vector_div(lhs: "Vector", rhs: "Vector") -> "Vector":
    ...

@define_op
def vector_exp(source: "Vector") -> "Vector":
    ...

@define_op
def vector_zeros(shape: Tuple[int, ...]) -> "Vector":
    ...

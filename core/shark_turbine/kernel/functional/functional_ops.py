from typing import (
    Any,
    List,
    Tuple,
    Optional,
    Iterator,
    overload,
    Callable,
    Tuple,
)
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Index, Vector

from ..ops.base import (
    define_op,
)

__all__ = ["memory_to_register", "mma", "memory_getitem", "memory_setitem"]


@define_op
def memory_getitem(kernel_buffer, key) -> "Memory":
    ...


@define_op
def memory_setitem(kernel_buffer, key, item) -> None:
    ...

@define_op
def memory_to_register(memory: "Memory") -> "Register":
    ...


@define_op
def mma(lhs: "Register", rhs: "Register", acc: "Register") -> "Register":
    ...

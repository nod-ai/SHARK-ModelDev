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

__all__ = ["read", "write", "mma", "memory_getitem", "memory_setitem", "register_getitem", "register_setitem"]


@define_op
def memory_getitem(memory, key) -> "Memory":
    ...

@define_op
def memory_setitem(memory, key, item) -> None:
    ...

@define_op
def register_getitem(register, key) -> "Register":
    ...

@define_op
def register_setitem(register, key, item) -> None:
    ...

@define_op
def memory_setitem(memory, key, item) -> None:
    ...

@define_op
def read(memory: "Memory") -> "Register":
    ...

@define_op
def write(register: "Register", memory: "Memory") -> None:
    ...

@define_op
def mma(lhs: "Register", rhs: "Register", acc: "Register") -> "Register":
    ...

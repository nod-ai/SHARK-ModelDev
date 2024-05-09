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

__all__ = [
    "construct_register_from_metadata",
    "read",
    "write",
    "mma",
    "memory_getitem",
    "memory_setitem",
    "register_getitem",
    "register_setitem",
    "tiled_loop",
]


@define_op
def memory_getitem(memory, key) -> "Memory": ...


@define_op
def memory_setitem(memory, key, item) -> None: ...


@define_op
def register_getitem(register, key) -> "Register": ...


@define_op
def register_setitem(register, key, item) -> None: ...


@define_op
def construct_register_from_metadata(shape, dtype, value) -> None: ...


@define_op
def memory_setitem(memory, key, item) -> None: ...


@define_op
def read(memory: "Memory", elements_pre_thread) -> "Register": ...


@define_op
def write(register: "Register", memory: "Memory", elements_pre_thread) -> None: ...


@define_op
def mma(lhs: "Register", rhs: "Register", acc: "Register") -> "Register": ...


@define_op
def tiled_loop(axis: "IndexExpr", init_args): ...


# Ops used in the codegen
@define_op
def alloc_shared(shape, dtype): ...


@define_op
def barrier(): ...


@define_op
def sync(value): ...


@define_op
def get_result(value, index): ...


@define_op
def read_shared(memory: "Memory", elements_pre_thread) -> "Register": ...


@define_op
def write_shared(
    register: "Register", memory: "Memory", elements_pre_thread
) -> None: ...

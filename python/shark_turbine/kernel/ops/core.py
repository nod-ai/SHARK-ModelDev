from typing import Any

from torch import fx

from .base import (
    define_op,
)

__all__ = [
    "kernel_buffer_getitem",
    "kernel_buffer_setitem",
    "thread_program_id",
]


@define_op
def kernel_buffer_getitem(kernel_buffer, key) -> Any:
    ...


@define_op
def kernel_buffer_setitem(kernel_buffer, key, item) -> None:
    ...


@define_op
def thread_program_id() -> int:
    ...

from torch import fx

from .base import (
    define_op,
)

__all__ = [
    "thread_program_id",
]

@define_op
def thread_program_id() -> int:
    ...
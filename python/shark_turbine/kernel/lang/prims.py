from typing import assert_type

from .. import ops

from .._support.tracing import (
    BaseContext,
    CompiledContext,
    custom_primitive_fn,
    eager_context,
)

__all__ = [
    "is_debug",
    "program_id",
]


def is_debug() -> bool:
    """Returns whether we are currently executing a kernel in eager debug mode."""
    return BaseContext.current().eager

program_id = ops.thread_program_id

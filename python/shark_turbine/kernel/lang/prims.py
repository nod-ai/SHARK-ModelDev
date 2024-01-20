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
    "constant",
    "dot",
    "for_loop",
    "load",
    "store",
]


def is_debug() -> bool:
    """Returns whether we are currently executing a kernel in eager debug mode."""
    return BaseContext.current().eager


# Core language operations
program_id = ops.thread_program_id

# Math Operations
constant = ops.vector_constant

# Reduction Operations
dot = ops.vector_dot

# Control Flow Operations
for_loop = ops.for_loop

# Memory Operations
load = ops.kernel_buffer_load
store = ops.kernel_buffer_store

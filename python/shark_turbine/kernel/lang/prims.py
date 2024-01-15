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
    "add",
    "sub",
    "mul",
    "div",
    "exp",
    "constant",
    "max",
    "sum",
    "dot",
    "for_loop",
    "load",
    "store"
]


def is_debug() -> bool:
    """Returns whether we are currently executing a kernel in eager debug mode."""
    return BaseContext.current().eager


program_id = ops.thread_program_id

add = ops.vector_add
sub = ops.vector_sub
mul = ops.vector_mul
div = ops.vector_div
exp = ops.vector_exp
constant = ops.vector_constant

max = ops.vector_max
sum = ops.vector_sum
dot = ops.vector_dot

for_loop = ops.for_loop

load = ops.kernel_buffer_load
store = ops.kernel_buffer_store

from typing import Optional

import functools
import math

import torch.fx as fx

from ..lang import (
    GlobalBuffer,
    Grid,
)

from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    EagerContext,
    KernelTracer,
)

__all__ = [
    "thread",
]


def thread(f=None, *, eager: bool = False):
    if f is None:
        return functools.partial(thread, eager=eager)

    # Eagerly capture the trace and attach it to the wrapped function.
    tracer = KernelTracer()
    with CompiledContext(tracer) as context:
        g = tracer.trace(f)
        gm = fx.GraphModule(tracer.root, g, f.__name__)

    @functools.wraps(f)
    def wrapped(*args, grid: Optional[Grid] = None, **kwargs):
        # Detect partial application.
        if not args:
            assert not kwargs, (
                f"Partial application can only have "
                f"a 'grid' kwarg. Has: {kwargs.keys()}"
            )
            return functools.partial(wrapped, *args, grid=grid, **kwargs)

        # Check parameters.
        if grid is None:
            raise ValueError("grid= is required for invoking a block_kernel")

        if eager:
            _eager_execute(f, *args, grid=grid, **kwargs)
        else:
            assert False, "Calling compiled not yet supported"

    wrapped.tk_trace = CapturedTrace(gm)
    return wrapped


def _eager_execute(f, *args, grid: Grid, **kwargs):
    rank = len(grid)
    with EagerContext(rank=rank) as context:
        # Transform args to KernelBuffers.
        buffer_args = [
            arg if isinstance(arg, GlobalBuffer) else GlobalBuffer(arg) for arg in args
        ]
        volume = math.prod(grid)
        current_thread = context.current_thread
        for it in range(volume):
            for i in range(rank - 1):
                current_thread[i] = it // grid[i]
                it = it % grid[i]
            current_thread[-1] = it
            f(*buffer_args, **kwargs)

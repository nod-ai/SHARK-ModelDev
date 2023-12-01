from typing import Generic, Optional, TypeVar, Callable, Union, assert_type, cast

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
    Launchable,
)

__all__ = [
    "thread",
]

TCallable = TypeVar("TCallable", bound=Callable)


def thread(f: TCallable) -> TCallable:
    # Eagerly capture the trace and attach it to the wrapped function.
    tracer = KernelTracer()
    with CompiledContext(tracer) as context:
        g = tracer.trace(f)
        gm = fx.GraphModule(tracer.root, g, f.__name__)

    return UnconfiguredThread[TCallable](f.__name__, f, CapturedTrace(gm))


class UnconfiguredThread(Generic[TCallable]):
    def __init__(self, name: str, wrapped_f: TCallable, trace: CapturedTrace):
        self._name = name
        self._wrapped_f = wrapped_f
        self._trace = trace

    def __getitem__(self, grid: Union[int, Grid]) -> TCallable:
        if isinstance(grid, int):
            grid = (grid,)
        assert isinstance(grid, tuple) and all(isinstance(i, int) for i in grid)
        return cast(
            TCallable, LaunchableThread(grid, self._name, self._wrapped_f, self._trace)
        )

    def __repr__(self):
        return f"tk.gen.thread @{self._name}[no grid]"


class LaunchableThread(Launchable):
    def __init__(
        self, grid: Grid, name: str, eager_function: Callable, trace: CapturedTrace
    ):
        super().__init__(eager_function)
        self.grid = grid
        self._name = name
        self._trace = trace

    def eager_execute(self, args, kwargs):
        grid = self.grid
        rank = len(grid)
        with EagerContext(rank=rank) as context:
            # Transform args to KernelBuffers.
            buffer_args = [
                arg if isinstance(arg, GlobalBuffer) else GlobalBuffer(arg)
                for arg in args
            ]
            volume = math.prod(grid)
            current_thread = context.current_thread
            for it in range(volume):
                for i in range(rank - 1):
                    current_thread[i] = it // grid[i]
                    it = it % grid[i]
                current_thread[-1] = it
                self._eager_function(*buffer_args, **kwargs)

    def __repr__(self):
        return f"tk.gen.thread @{self._name}[{', '.join(str(i) for i in self.grid)}]"

from typing import Generic, Optional, Type, TypeVar, Callable, Union, assert_type, cast

import inspect
import math

import torch.fx as fx

from ..lang import (
    KernelBuffer,
    Grid,
    SymbolDef,
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


def thread(*symbolic_shape: SymbolDef):
    GridType = Grid[symbolic_shape]

    def decorator(f: Optional[TCallable] = None) -> "UnconfiguredThread[TCallable]":
        # Eagerly capture the trace and attach it to the wrapped function.
        tracer = KernelTracer()
        with CompiledContext(tracer) as context:
            g = tracer.trace(f)
            gm = fx.GraphModule(tracer.root, g, f.__name__)

        return UnconfiguredThread[TCallable](GridType, f.__name__, f, CapturedTrace(gm))

    return decorator


class UnconfiguredThread(Generic[TCallable]):
    def __init__(
        self,
        grid_type: Type[Grid],
        name: str,
        wrapped_f: TCallable,
        trace: CapturedTrace,
    ):
        self.grid_type = grid_type
        self._name = name
        self._wrapped_f = wrapped_f
        self._trace = trace

    def __getitem__(self, grid: Union[int, tuple[int]]) -> TCallable:
        if not isinstance(grid, tuple):
            grid = (grid,)
        assert isinstance(grid, tuple) and all(isinstance(i, int) for i in grid)
        grid = self.grid_type(*grid)
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
        self.grid_type = type(grid)
        self._name = name
        self._trace = trace
        self._sig = inspect.signature(eager_function)

    def eager_execute(self, args, kwargs):
        grid = self.grid
        rank = grid.rank
        with EagerContext(rank=rank) as context:
            sig = self._sig
            bound = sig.bind(*args, *kwargs)
            bound.apply_defaults()
            # Transform args to KernelBuffers.
            for arg_name in list(bound.arguments.keys()):
                arg_value = bound.arguments[arg_name]
                param = sig.parameters[arg_name]
                param_type = param.annotation
                if isinstance(param_type, type) and issubclass(
                    param_type, KernelBuffer
                ):
                    kernel_buffer = param_type(arg_value)
                    bound.arguments[arg_name] = kernel_buffer
            volume = math.prod(grid)
            current_thread = context.current_thread
            for it in range(volume):
                for i in range(rank - 1):
                    current_thread[i] = it // grid[i]
                    it = it % grid[i]
                current_thread[-1] = it
                self._eager_function(*bound.args, **bound.kwargs)

    def __repr__(self):
        return f"tk.gen.thread @{self._name}[{', '.join(str(i) for i in self.grid)}]"

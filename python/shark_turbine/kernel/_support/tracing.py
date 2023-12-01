from typing import Optional, TypeVar, Callable, assert_type, cast

import functools
import threading

import torch.fx as fx

from ..lang.types import (
    GlobalBuffer,
    Grid,
)

_tls = threading.local()
TCallable = TypeVar("TCallable", bound=Callable)

###############################################################################
# Wrapped tracing trampolines for proxy objects.
# These only get called during tracing of proxy objects.
###############################################################################


@fx.wrap
def _global_buffer_setitem(kernel_buffer: GlobalBuffer, key, item) -> None:
    ...


###############################################################################
# Tracing machinery
###############################################################################


class GlobalBufferProxy(fx.Proxy):
    """Custom proxy for GlobalBuffer so that we can override special methods."""

    def __setitem__(self, key, item):
        _global_buffer_setitem(self, key, item)


class KernelTracer(fx.Tracer):
    """Custom Tracer for generating a trace of a kernel computation."""

    def proxy(self, node: fx.Node) -> fx.Proxy:
        if node.type == GlobalBuffer:
            return GlobalBufferProxy(node, self)
        return super().proxy(node)


class CapturedTrace:
    def __init__(self, gm: fx.GraphModule):
        self.gm = gm


###############################################################################
# Execution
###############################################################################


class BaseContext:
    def __init__(self, *, eager: bool):
        self.eager = eager

    @staticmethod
    def current() -> "BaseContext":
        try:
            return _tls.context[-1]
        except (AttributeError, IndexError):
            raise RuntimeError("No context is on the stack")

    def __enter__(self) -> "BaseContext":
        try:
            stack = _tls.context
        except AttributeError:
            stack = []
            _tls.context = stack
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tls.context.pop()


class EagerContext(BaseContext):
    def __init__(self, rank: int = 0):
        super().__init__(eager=True)
        self.rank = rank
        self.current_thread: list[int] = rank * [0]


class CompiledContext(BaseContext):
    def __init__(self, tracer: KernelTracer):
        super().__init__(eager=False)
        self.tracer = tracer


###############################################################################
# Helpers
###############################################################################


def eager_context() -> EagerContext:
    context = BaseContext.current()
    assert context.eager, "Expected to be executed against an EagerContext"
    assert_type(context, EagerContext)
    return context


def custom_primitive_fn(
    f: Optional[TCallable] = None, *, compiled: Callable
) -> TCallable:
    """Decorator for a primitive function with a custom callback for tracing.

    The wrapped function will be invoked as-is when executing eagerly. When
    tracing, the `compiled` callback will be invoked with the same signature
    but with the `CompiledContext` added as a first postional argument.
    """
    if f is None:
        return functools.partial(custom_primitive_fn, compiled=compiled)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):  # type: ignore
        context = BaseContext.current()
        if context.eager:
            return f(*args, **kwargs)
        else:
            assert_type(context, CompiledContext)
            return compiled(context, *args, **kwargs)

    return cast(TCallable, wrapper)

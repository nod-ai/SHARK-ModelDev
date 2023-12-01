from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Callable, assert_type, cast

import functools
import threading
import warnings

import torch.fx as fx

from ..lang.types import (
    KernelBuffer,
    Grid,
)

_tls = threading.local()
TCallable = TypeVar("TCallable", bound=Callable)

###############################################################################
# Wrapped tracing trampolines for proxy objects.
# These only get called during tracing of proxy objects.
###############################################################################


@fx.wrap
def _kernel_buffer_setitem(kernel_buffer: KernelBuffer, key, item) -> None:
    ...


###############################################################################
# Tracing machinery
###############################################################################


class KernelBufferProxy(fx.Proxy):
    """Custom proxy for KernelBuffer so that we can override special methods."""

    def __setitem__(self, key, item):
        _kernel_buffer_setitem(self, key, item)


class KernelTracer(fx.Tracer):
    """Custom Tracer for generating a trace of a kernel computation."""

    def proxy(self, node: fx.Node) -> fx.Proxy:
        if node.type == KernelBuffer:
            return KernelBufferProxy(node, self)
        return super().proxy(node)


class CapturedTrace:
    def __init__(self, gm: fx.GraphModule):
        self.gm = gm


###############################################################################
# Execution context.
# A valid BaseContext derived instance (EagerContext or CompiledContext) must
# be active for any evaluation of a generated/traced function.
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
# Launch context
# The launch context controls how the call into a kernel is dispatched.
# This can either be to run it eagerly for debugging or some higher order
# integration.
###############################################################################


class Launchable(ABC):
    """Base class for objects which behave like a kernel launch when called."""

    def __init__(self, eager_function: Callable):
        self._eager_function = eager_function

    def __call__(self, *args, **kwargs):
        launch_context = LaunchContext.current()
        return launch_context.launch(self, args, kwargs)

    @abstractmethod
    def eager_execute(self, args, kwargs):
        ...


class LaunchContext(ABC):
    @staticmethod
    def current() -> "LaunchContext":
        try:
            return _tls.launch[-1]
        except (AttributeError, IndexError):
            warnings.warn(
                "defaulting to debug/eager execution of tk kernel launch "
                "because no launch context has been established"
            )
            return DebugLaunchContext()

    def __enter__(self) -> "LaunchContext":
        try:
            stack = _tls.launch
        except AttributeError:
            stack = []
            _tls.launch = stack
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tls.launch.pop()

    @abstractmethod
    def launch(self, launchable: Launchable, args, kwargs):
        ...


class DebugLaunchContext(LaunchContext):
    def launch(self, launchable: Launchable, args, kwargs):
        return launchable.eager_execute(args, kwargs)


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

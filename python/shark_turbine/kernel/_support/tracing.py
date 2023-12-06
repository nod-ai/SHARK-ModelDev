from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Callable, Type, assert_type, cast

import functools
import warnings

import torch.fx as fx

from .indexing import (
    KernelBuffer,
)

from ..lang.types import (
    Index,
)

from .. import ops
from ..ops.base import (
    OpDispatcher,
)

from . import context

TCallable = TypeVar("TCallable", bound=Callable)

###############################################################################
# Tracing machinery
###############################################################################


class KernelBufferProxy(fx.Proxy):
    """Custom proxy for KernelBuffer so that we can override special methods."""

    def __init__(
        self, node: fx.Node, tracer: "KernelTracer", orig_type: Type[KernelBuffer]
    ):
        super().__init__(node, tracer)
        self._orig_type = orig_type
        # The shape and rank are statically available (not proxied).
        self.symbolic_shape = orig_type.symbolic_shape
        self.rank = orig_type.rank

    def __getitem__(self, key):
        return ops.kernel_buffer_getitem(self, key)

    def __setitem__(self, key, item):
        ops.kernel_buffer_setitem(self, key, item)


class KernelTracer(fx.Tracer):
    """Custom Tracer for generating a trace of a kernel computation."""

    def proxy(self, node: fx.Node) -> fx.Proxy:
        t = node.type
        if t is not None and issubclass(t, KernelBuffer):
            return KernelBufferProxy(node, self, t)
        return super().proxy(node)


class CapturedTrace:
    def __init__(self, gm: fx.GraphModule):
        self.gm = gm


###############################################################################
# Execution context.
# A valid BaseContext derived instance (EagerContext or CompiledContext) must
# be active for any evaluation of a generated/traced function.
###############################################################################


class BaseContext(OpDispatcher):
    __tk_context_idname__ = "ExecutionContext"

    def __init__(self, *, eager: bool):
        self.eager = eager

    @staticmethod
    def current() -> "BaseContext":
        return context.current(BaseContext)

    def __enter__(self) -> "BaseContext":
        context.push(OpDispatcher, self)
        return context.push(BaseContext, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pop(OpDispatcher, self)
        context.pop(BaseContext, self)


class EagerContext(BaseContext):
    def __init__(self, rank: int = 0):
        super().__init__(eager=True)
        self.rank = rank
        self.current_thread: list[int] = rank * [0]

    def handle_thread_program_id(self, op, axis: int) -> int:
        assert axis >= 0 and axis < self.rank
        return Index(self.current_thread[axis])

    def handle_kernel_buffer_getitem(self, op, kernel_buffer: KernelBuffer, key):
        return kernel_buffer._tensor.__getitem__(key)

    def handle_kernel_buffer_setitem(self, op, kernel_buffer: KernelBuffer, key, item):
        kernel_buffer._tensor.__setitem__(key, item)


class CompiledContext(BaseContext):
    def __init__(self, tracer: KernelTracer):
        super().__init__(eager=False)
        self.tracer = tracer

    def handle_thread_program_id(self, op, axis: int) -> Index:
        proxy = self.tracer.create_proxy(
            "call_function", op, args=(axis,), kwargs={}, type_expr=Index
        )
        return proxy

    def handle_kernel_buffer_getitem(self, op, kernel_buffer: KernelBuffer, key):
        return self.tracer.create_proxy(
            "call_function",
            op,
            args=(kernel_buffer, key),
            kwargs={},
        )

    def handle_kernel_buffer_setitem(self, op, kernel_buffer: KernelBuffer, key, item):
        self.tracer.create_proxy(
            "call_function",
            target=op,
            args=(kernel_buffer, key, item),
            kwargs={},
        )


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
    __tk_context_idname__ = "ExecutionContext"

    @staticmethod
    def current() -> "LaunchContext":
        try:
            return context.current(LaunchContext)
        except IndexError:
            warnings.warn(
                "defaulting to debug/eager execution of tk kernel launch "
                "because no launch context has been established"
            )
            return DebugLaunchContext()

    def __enter__(self) -> "LaunchContext":
        return context.push(LaunchContext, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pop(LaunchContext, self)

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

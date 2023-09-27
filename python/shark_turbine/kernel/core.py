# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import assert_type, List, Optional, Tuple

import functools
import math
import threading

import torch
import torch.fx as fx
from torch import SymInt

_tls = threading.local()

###############################################################################
# Language
###############################################################################

Grid = Tuple[int, ...]


class KernelBuffer:
    """Represents input and output buffers to a kernel."""

    __slots__ = [
        "_tensor",
    ]

    def __init__(self, tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor but got {tensor}"
        self._tensor = tensor

    def __repr__(self):
        return f"KernelBuffer({self._tensor})"

    def __setitem__(self, key, item):
        self._tensor.__setitem__(key, item)


def program_id(axis: int) -> int:
    context = BaseContext.current()
    if context.eager:
        # Eager.
        assert_type(context, EagerContext)
        assert axis >= 0 and axis < context.rank
        return context.current_thread[axis]
    else:
        # Compiled. Note that tracing must be open coded on this
        # function because it does not take a proxy as an argument
        # (and therefore, the symbolic tracer exempts it from tracing
        # according to its heuristic).
        assert_type(context, CompiledContext)
        proxy = context.tracer.create_proxy("call_function", program_id, (axis,), {})
        return proxy


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
    """Custom Tracer for generating a trace of a kernel.

    A "kernel" in this context takes as input KernelBuffer objects
    for all inputs and outputs. In eager mode, these are just regular
    Tensors where we read slices from the inputs and write slices to
    the outputs. In traced mode, we substitute appropriate proxies
    to handle the limited operations that they support.
    """

    def proxy(self, node: fx.Node) -> fx.Proxy:
        if node.type == KernelBuffer:
            return KernelBufferProxy(node, self)
        return super().proxy(node)


class CapturedTrace:
    def __init__(self, gm: fx.GraphModule):
        self.gm = gm


###############################################################################
# Decorators
###############################################################################


def block_kernel(f=None, *, eager: bool = False):
    if f is None:
        return functools.partial(block_kernel, eager=eager)

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
            _eager_execute_kernel(f, *args, grid=grid, **kwargs)
        else:
            assert False, "Calling compiled not yet supported"

    wrapped.tk_trace = CapturedTrace(gm)
    return wrapped


###############################################################################
# Execution
###############################################################################


class BaseContext:
    def __init__(self, eager: bool):
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
        super().__init__(True)
        self.rank = rank
        self.current_thread: List[int] = rank * [0]


class CompiledContext(BaseContext):
    def __init__(self, tracer: KernelTracer):
        super().__init__(False)
        self.tracer = tracer


def _eager_execute_kernel(f, *args, grid: Grid, **kwargs):
    rank = len(grid)
    with EagerContext(rank=rank) as context:
        # Transform args to KernelBuffers.
        buffer_args = [
            arg if isinstance(arg, KernelBuffer) else KernelBuffer(arg) for arg in args
        ]
        volume = math.prod(grid)
        current_thread = context.current_thread
        for it in range(volume):
            for i in range(rank - 1):
                current_thread[i] = it // grid[i]
                it = it % grid[i]
            current_thread[-1] = it
            f(*buffer_args, **kwargs)

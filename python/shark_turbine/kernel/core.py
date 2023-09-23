# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Optional, Tuple

import functools
import math
import threading

import torch.fx

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
    context = Context.current()
    if context.executing_eagerly:
        assert axis >= 0 and axis < context.rank
        return context.current_thread[axis]

    assert False, "Non eager not yet implemented"


###############################################################################
# Tracing machinery
###############################################################################


class KernelBufferProxy(torch.fx.Proxy):
    """Custom proxy for KernelBuffer so that we can override special methods."""

    ...


class KernelTracer(torch.fx.Tracer):
    """Custom Tracer for generating a trace of a kernel.

    A "kernel" in this context takes as input KernelBuffer objects
    for all inputs and outputs. In eager mode, these are just regular
    Tensors where we read slices from the inputs and write slices to
    the outputs. In traced mode, we substitute appropriate proxies
    to handle the limited operations that they support.
    """

    def proxy(self, node: torch.fx.Node) -> torch.fx.Proxy:
        if node.type == KernelBuffer:
            return KernelBufferProxy(node, self)
        return super().proxy(node)


###############################################################################
# Decorators
###############################################################################


def block_kernel(f=None, *, eager: bool = False):
    if f is None:
        return functools.partial(block_kernel, eager=eager)

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
            assert False, "Compiled not yet supported"

    return wrapped


###############################################################################
# Execution
###############################################################################


class Context:
    def __init__(self, executing_eagerly: bool = False, rank: int = 0):
        self.rank = rank
        self.current_thread: List[int] = rank * [0]
        self.executing_eagerly = executing_eagerly

    @staticmethod
    def current() -> "Context":
        try:
            return _tls.context[-1]
        except (AttributeError, IndexError):
            raise RuntimeError("No eager context is on the stack")

    def __enter__(self) -> "Context":
        try:
            stack = _tls.context
        except AttributeError:
            stack = []
            _tls.context = stack
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tls.context.pop()


def _eager_execute_kernel(f, *args, grid: Grid, **kwargs):
    rank = len(grid)
    with Context(executing_eagerly=True, rank=rank) as context:
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

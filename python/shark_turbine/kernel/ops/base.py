"""Support for defining the op library and dispatch."""

from typing import Callable, Type, TypeVar

import functools

T = TypeVar("T")

class OpDispatcher:
    ...


def define_op(dispatcher_ty: Type[OpDispatcher], idname: str) -> Callable[[T], T]:
    def decorator(f: T) -> T:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            ...

        wrapped.__tk_op_idname__ = idname
        return wrapped
    
    return decorator
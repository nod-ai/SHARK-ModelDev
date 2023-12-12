from typing import Type


__all__ = [
    "Index",
]

###############################################################################
# Index and specific sized integer types
###############################################################################


def _impl_fixed_int(t: Type[int]):
    """Mixes in dunder functions for integer math to an `int` derived type.

    The result of the computation will be cast to type `t` before returning.
    """
    t.__add__ = lambda a, b: t(super(t, a).__add__(b))
    t.__sub__ = lambda a, b: t(super(t, a).__sub__(b))
    t.__mul__ = lambda a, b: t(super(t, a).__mul__(b))
    t.__floordiv__ = lambda a, b: t(super(t, a).__floordiv__(b))
    t.__mod__ = lambda a, b: t(super(t, a).__mod__(b))
    t.__pow__ = lambda a, b, modulo=None: t(super(t, a).__pow__(b, modulo))
    t.__pos__ = lambda a: t(super(t, a).__pos__())
    t.__neg__ = lambda a: t(super(t, a).__neg__())
    return t


@_impl_fixed_int
class Index(int):
    """An index type that is isomorphic to MLIR `index`.

    At the Python level, this is just an int.
    """

    ...

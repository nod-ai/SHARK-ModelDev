from typing import Type

from ..ops.math import vector_add, vector_sub, vector_mul, vector_div


__all__ = [
    "Index",
    "Vector",
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


class Vector:
    def __add__(self, other: "Vector") -> "Vector":
        return vector_add(self, other)

    def __sub__(self, other: "Vector") -> "Vector":
        return vector_sub(self, other)

    def __mul__(self, other: "Vector") -> "Vector":
        return vector_mul(self, other)

    def __truediv__(self, other: "Vector") -> "Vector":
        return vector_div(self, other)

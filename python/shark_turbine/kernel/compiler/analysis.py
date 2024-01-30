from typing import Any, Optional, Type, Union

import torch.fx as fx

from .base import CodegenError

from .._support.indexing import (
    index_expr,
    IndexingContext,
    IndexExpr,
)


NormalizedSlice = Union[slice, None, IndexExpr]


def _norm_slice_spec(rank: int, slice_spec) -> list[NormalizedSlice]:
    def _norm_single_slice(s):
        if s is None or s is ...:
            return s
        if isinstance(s, slice):
            start = s.start
            stop = s.stop
            step = s.step
            # Set defaults.
            if start is None:
                start = 0
            if step is None:
                step = 1

            return slice(
                index_expr(start),
                stop if stop is None else index_expr(stop),
                index_expr(step),
            )
        else:
            # Index extraction case.
            return index_expr(s)

    if not isinstance(slice_spec, tuple):
        slice_spec = (slice_spec,)
    norm_slices = [_norm_single_slice(s) for s in slice_spec]

    # Replace any ellipses with rank-filling None values.
    none_count = norm_slices.count(None)
    ellipses_count = norm_slices.count(...)
    if ellipses_count == 1:
        # Expand by the original list of slices less any unit dim insertions.
        # If negative, this does nothing and will be caught later upon
        # rank validation.
        expand_index = norm_slices.index(...)
        del norm_slices[expand_index]
        expansion_count = (rank + none_count) - len(norm_slices)
        for _ in range(expansion_count):
            norm_slices.insert(expand_index, slice(None))
    elif ellipses_count > 1:
        raise IndexError(
            f"Cannot index into a rank expanding referrent with multiple `...` values"
        )
    return norm_slices


def _norm_slice_axis(
    idxc: IndexingContext,
    axis_bound_expr: IndexExpr,
    s: NormalizedSlice,
    allow_reverse_step: bool,
    allow_non_unit_step: bool,
) -> Optional[slice]:
    if s is None:
        return s
    if isinstance(s, slice):
        # Fall-through to normalize a full slice
        pass
    else:
        # Assume that it is an index extract of the axis.
        s = idxc.simplify_expr(s)
        nn_assump = s.is_nonnegative
        if nn_assump is None:
            raise IndexError(
                f"Indices used in code generation must be statically negative or postive. "
                f"Got: {s}"
            )
        if nn_assump:
            return s
        else:
            # Index from end.
            return axis_bound_expr + s

    start = idxc.simplify_expr(s.start)
    # If stop is None, then it takes the value of the axis bound.
    stop = (
        idxc.simplify_expr(axis_bound_expr)
        if s.stop is None
        else idxc.simplify_expr(s.stop)
    )
    step = idxc.simplify_expr(s.step)

    # Validate direction and step.
    if not allow_non_unit_step:
        if not (step == 1):
            raise IndexError(f"Non-unit step not allowed in this context. Got: {step}")
    elif not allow_reverse_step and not step.is_positive:
        raise IndexError(f"Reverse step not allowed in this context. Got: {step}")

    return slice(start, stop, step)


class SliceAnalysis:
    """Analyses Python slicing notations such that it can be validated and code generated.

    The numpy page has a good description here:
        https://numpy.org/doc/1.26/user/basics.indexing.html

    This class analyzes:
      * Basic Indexing
      * Slicing and Striding
      * Dimensional Indexing Tools

    Note that `None` is interpreted as `np.newaxis` (which we do not support).

    Each element of a slice specification can be:
      * An arbitrary Python value representing a single element span
      * None to indicate a new unit dimension
      * Ellipses to indicate space filling `slice()`
      * A `slice` object

    Such a specification is decomposed into a `source_slice` which does not
    include any rank broadcasting and a `broadcast_slice` which includes
    any rank expansion. Depending on the operation being code generated,
    these will be handled differently. All loose Python values will
    be promoted into a `slice` object.

    Raises:
      IndexError on any violations of Python indexing semantics which
      are statically determined during analysis.
    """

    def __init__(
        self,
        ref: tuple[IndexExpr, ...],
        slice_spec,
        *,
        allow_reverse_step: bool = False,
        allow_non_unit_step: bool = False,
    ):
        self.ref = ref
        norm_slices = _norm_slice_spec(len(ref), slice_spec)

        # Compute an expanded version of ref that has None values for
        # any to-be-inserted unit dims. This will be the same size
        # as slices.
        self.expanded_ref: list[Optional[IndexExpr]] = list(ref)
        for i in (i for i, entry in enumerate(norm_slices) if entry is None):
            self.expanded_ref.insert(i, None)

        rank = len(norm_slices)
        assert len(self.expanded_ref) == rank

        # Normalizes values for any None ranges.
        self.slices = [None] * rank
        idxc = IndexingContext.current()
        for i in range(rank):
            axis_bound_expr = self.expanded_ref[i]
            self.slices[i] = _norm_slice_axis(
                idxc,
                axis_bound_expr,
                norm_slices[i],
                allow_reverse_step=allow_reverse_step,
                allow_non_unit_step=allow_non_unit_step,
            )

    def __repr__(self):
        return repr(self.slices)

    @property
    def symbolic_shape(self) -> tuple[Union[int, IndexExpr]]:
        """Resolves the symbolic shape of the result of this slice.

        Forces symbolic normalization if it has not already been done.
        Any rank broadcast dimensions will be retained as None.
        """

        def _item(s: Optional[slice]):
            if s is None:
                return None
            ctx = IndexingContext.current()
            # Detect special unit 0-step slices.
            if not isinstance(s, slice):
                # Index extraction.
                return 1
            start = s.start
            stop = s.stop
            step = s.step

            result = ((stop - start) // step).simplify()
            return result

        return [_item(s) for s in self.slices]

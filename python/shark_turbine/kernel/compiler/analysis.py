from typing import Any, Optional, Type, Union

import torch.fx as fx

from .base import CodegenError

from .._support.indexing import (
    BoundedSymbolicValue,
    IndexingContext,
    SymbolDef,
    SymbolExpr,
)


NormalizedSlice = list[Union[slice, None]]


def _symbolize_slice_value(value):
    # TODO: I don't like this and wish this happened more automatically somehow.
    if isinstance(value, fx.Node):
        sym_type = value.type
        if sym_type and issubclass(sym_type, BoundedSymbolicValue):
            return sym_type()
        return value
    else:
        return value


def _norm_slice_spec(rank: int, slice_spec) -> NormalizedSlice:
    def _norm_single_slice(s):
        if s is None or s is ...:
            return s
        if isinstance(s, slice):
            # Validate.
            if s.step == 0:
                # A zero step is illegal, but we use it to signal an integer index
                # vs a range.
                raise IndexError(f"slice with step 0 is illegal (got {s})")
            return s
        else:
            # Promote a raw value to our special 0-step slice.
            return slice(s, 0, 0)

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

    def __init__(self, ref: tuple[SymbolExpr, ...], slice_spec):
        self.ref = ref
        self.slices = _norm_slice_spec(len(ref), slice_spec)

        # Compute an expanded version of ref that has None values for
        # any to-be-inserted unit dims. This will be the same size
        # as slices.
        self.expanded_ref: list[Optional[SymbolExpr]] = list(ref)
        for i in (i for i, entry in enumerate(self.slices) if entry is None):
            self.expanded_ref.insert(i, None)
        assert len(self.expanded_ref) == len(self.slices)
        self._is_symbolic_normalized = False

    def __repr__(self):
        return repr(self.slices)

    def normalize_symbolic_ranges(
        self, *, allow_reverse_step: bool = False, allow_non_unit_step: bool = False
    ):
        """Uses the IndexingContext to normalize range for any None fields.

        This fully populates the fields of every slice with either
        integers or SymbolExprs. Does not modify slice fields that are
        non-None.

        Raises IndexError for any variations that are not supported
        or cannot be statically derived.
        """
        if self._is_symbolic_normalized:
            return

        def norm(dim_expr: SymbolExpr, s: Optional[slice]) -> Optional[slice]:
            if s is None:
                return s
            ctx = IndexingContext.current()
            start = s.start
            stop = s.stop
            step = s.step

            # Set defaults.
            if start is None:
                start = 0
            if stop is None:
                stop = dim_expr
            if step is None:
                step = 1

            # Symbolize for analysis.
            start_sym = _symbolize_slice_value(start)
            stop_sym = _symbolize_slice_value(stop)
            step_sym = _symbolize_slice_value(step)

            # Evaluate facts for start.
            if isinstance(start_sym, SymbolExpr):
                start_is_non_negative = start_sym.is_non_negative()
            elif isinstance(start_sym, int):
                start_is_non_negative = start_sym >= 0
            else:
                raise IndexError(
                    f"A symbolically evaluable start index is required (got: {start_sym} (type {type(start_sym)}))"
                )

            # Evaluate facts for stop.
            if isinstance(stop_sym, SymbolExpr):
                stop_is_non_negative = stop_sym.is_non_negative()
                stop_is_zero = False
            elif isinstance(stop_sym, int):
                stop_is_non_negative = stop_sym >= 0
                stop_is_zero = stop_sym == 0
            else:
                raise IndexError(
                    f"A symbolically evaluable stop index is required (got: {stop_sym} (type {type(stop_sym)}))"
                )

            # Evaluate facts for step.
            if isinstance(step_sym, SymbolExpr):
                reverse_step = step_sym.is_negative()
                unit_step = step_sym.is_one()
                zero_step = False
            elif isinstance(step_sym, int):
                reverse_step = step_sym < 0
                unit_step = step_sym == 1
                zero_step = step_sym == 0
            else:
                raise IndexError(
                    f"A symbolically evaluable step is required (got: {step_sym} (type {type(step_sym)}))"
                )

            # Validate step constraints.
            if zero_step:
                # This is our special marker for a unit (non-range extract).
                assert (
                    stop_is_zero
                ), "slices of non zero stop and zero step should have been filtered"
            else:
                if not allow_non_unit_step and not unit_step:
                    raise IndexError(
                        f"Only unit steps are supported in this context (got slice({start_sym}, {stop_sym}, {step_sym}))"
                    )

                if not allow_reverse_step and reverse_step:
                    raise IndexError(
                        f"Only forward steps are supported in this context (got slice({start_sym}, {stop_sym}, {step_sym}))"
                    )

            # Normalize negative start/stop.
            if not start_is_non_negative:
                raise IndexError(f"NYI: Negative slice start")
            if not stop_is_non_negative:
                raise IndexError(f"NYI: Negative slice stop")

            return slice(start, stop, step)

        for i in range(len(self.slices)):
            expr = self.expanded_ref[i]
            self.slices[i] = norm(expr, self.slices[i])
        self._is_symbolic_normalized = True

    @property
    def symbolic_shape(self) -> tuple[Union[int, SymbolExpr]]:
        """Resolves the symbolic shape of the result of this slice.

        Forces symbolic normalization if it has not already been done.
        Any rank broadcast dimensions will be retained as None.
        """
        self.normalize_symbolic_ranges()

        def _item(s: Optional[slice]):
            if s is None:
                return None
            ctx = IndexingContext.current()
            # Detect special unit 0-step slices.
            if s.stop == 0 and s.step == 0:
                return 1
            start = s.start
            stop = s.stop

            # TODO: This is a hack to work around that I don't have the full
            # symbolic expression support in yet. We should just be asking
            # the symbols to evaluate.
            if isinstance(start, SymbolExpr):
                static_start = ctx.get_static_value(start)
            elif isinstance(start, int):
                static_start = start
            if isinstance(stop, SymbolExpr):
                static_stop = ctx.get_static_value(stop)
            elif isinstance(stop, int):
                static_stop = stop
            if static_start is not None and static_stop is not None:
                return static_stop - static_start

            raise IndexError(f"NYI: Non-statically resolved symbolic shapes")

        return [_item(s) for s in self.slices]

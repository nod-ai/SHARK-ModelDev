"""Code generation for generating vector-dialect based kernels.

Such kernels operate on global memory at the boundary, scheduling
actual loads/stores/computes to local vectors using PyTorch tensor
level operations executed as threads over a grid.
"""
from typing import Any, Callable, Type, Optional, Sequence, Union, List

from dataclasses import dataclass
import inspect
import operator as py_operator

import torch
import torch.fx as fx
import torch.utils._pytree as pytree

from .._support.indexing import (
    Grid,
    IndexingContext,
    KernelBuffer,
    SymbolDef,
    is_kernel_buffer_meta_derived,
)

from .._support.tracing import CapturedTrace

from .. import lang as tkl

from ..lang import (
    Index,
)

from .. import ops

from .analysis import (
    SliceAnalysis,
)

from .builder import (
    ModuleBuilder,
    ScalarBuilder,
)

from .base import (
    CodegenError,
    ValidationError,
)

from .ir import (
    AffineMap,
    Attribute,
    AffineExpr,
    AffineMapAttr,
    ArrayAttr,
    FunctionType,
    VectorType,
    DenseElementsAttr,
    F32Type,
    FloatAttr,
    InsertionPoint,
    IrType,
    Location,
    MemRefType,
    ShapedType,
    Value,
    VectorType,
    arith_d,
    func_d,
    math_d,
    vector_d,
    scf_d,
)

from .kernel_codegen import (
    BoundKernelSignature,
)

from . import op_matchers

ArgTypeUnion = Union[SymbolDef, Type[KernelBuffer]]


@dataclass
class NodeAttrs:
    # By default, integers are assumed signed. We propagate unsigned as graph
    # node attrs.
    unsigned: bool = False

    @staticmethod
    def load(py_value) -> "NodeAttrs":
        if isinstance(py_value, fx.Node):
            return NodeAttrs(unsigned=bool(py_value.meta.get("unsigned")))
        return NodeAttrs()

    def store(self, node: fx.Node):
        node.meta["unsigned"] = self.unsigned


class ThreadEmitter:
    """Emits a 'thread function' as a `func` with a signature derived from the gm."""

    OP_HANDLERS: dict[Any, Callable[["ThreadEmitter", fx.Node], None]] = {}

    def __init__(self, root_sig: BoundKernelSignature, trace: CapturedTrace):
        self._node_values: dict[fx.Node, List[Value]] = {}
        self._grid_axis_values: dict[int, Value] = {}
        self._root_sig = root_sig
        self.trace = trace
        self.ip = InsertionPoint(root_sig.entry_block)

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        values = self._node_values.get(node)
        if values is None:
            values = [self._root_sig.resolve_by_reference(("node", node))]
            self._node_values[node] = values
        return values

    def lookup_grid_axis_value(self, grid_axis: int) -> Value:
        value = self._grid_axis_values.get(grid_axis)
        if value is None:
            try:
                value = self._root_sig.resolve_by_reference(("grid", grid_axis))
            except KeyError:
                raise CodegenError(f"Grid axis {grid_axis} out of bounds")
            self._node_values[grid_axis] = [value]
        return value

    def bind_node_result(
        self, node: fx.Node, value: Value, *, attrs: Optional[NodeAttrs] = None
    ):
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = [value]

    def bind_node_results(
        self,
        node: fx.Node,
        values: List[Value],
        *,
        attrs: Optional[NodeAttrs] = None,
    ):
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = values

    def emit(self):
        with self.ip, Location.unknown():
            self.emit_graph(self.trace.get_root_graph())

    def emit_function_call_node(self, node: fx.Node):
        target_op = node.target
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")
        handler(self, node)
        # dump

    def emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function":
                self.emit_function_call_node(node)
            if node.op == "output":
                return node.args

    def emit_subgraph(self, subgraph: fx.Graph, implicit_capture: list[fx.Node]):
        # Map subgraph freevars -> implicit_capture
        freevars = self.trace.region_graph.inner_freevars[subgraph]
        assert len(freevars) == len(
            implicit_capture
        ), f"Expected {len(freevars)} implicit capture args, got {len(implicit_capture)}"
        for freevar, arg in zip(freevars, implicit_capture):
            self._node_values[freevar.node] = self.lookup_node_values(arg)

        # Emit subgraph
        return self.emit_graph(subgraph)

    def finish(self):
        with self.ip, Location.unknown():
            func_d.ReturnOp([])


def handle_op(op):
    def decorator(f: Callable[["ThreadEmitter", fx.Node], None]):
        ThreadEmitter.OP_HANDLERS[op] = f
        return None

    return decorator


###############################################################################
# Python/scalar ops
###############################################################################


@handle_op(py_operator.getitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        proxy, index = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguements") from e

    if not isinstance(proxy, fx.Node):
        raise CodegenError(f"Expected fx.Node")
    node_values = emitter.lookup_node_values(proxy)

    emitter.bind_node_result(node, node_values[index])


BINARY_ARITHMETIC_OPS = [
    (py_operator.add, "add"),
    (py_operator.mul, "mul"),
    (py_operator.sub, "sub"),
    (py_operator.mod, "mod"),
    (py_operator.floordiv, "floordiv"),
    (py_operator.truediv, "truediv"),
]


def binary_broadcast(lhs: Value, rhs: Value) -> tuple[bool, Value, Value]:
    lhs_type = lhs.type
    rhs_type = rhs.type
    lhs_is_vector = VectorType.isinstance(lhs_type)
    rhs_is_vector = VectorType.isinstance(rhs_type)
    if not lhs_is_vector and not rhs_is_vector:
        # Not vectors: return as-is.
        return False, lhs, rhs

    # Promote to vector.
    if not lhs_is_vector:
        lhs = vector_d.splat(VectorType([], lhs_type), lhs)
    if not rhs_is_vector:
        rhs = vector_d.splat(VectorType([], rhs_type), rhs)
    lhs_type = VectorType(lhs.type)
    rhs_type = VectorType(rhs.type)

    broadcast_shape = lhs_type.shape
    rhs_shape = rhs_type.shape
    rank = max(len(broadcast_shape), len(rhs_shape))
    while len(broadcast_shape) < rank:
        broadcast_shape.insert(0, 1)
    while len(rhs_shape) < rank:
        rhs_shape.insert(0, 1)

    for i in range(rank):
        a = broadcast_shape[i]
        b = rhs_shape[i]
        if a != b:
            if a != 1 and b != 1:
                raise CodegenError(
                    f"Binary operands are not broadcast compatible: {lhs_type}, {rhs_type}"
                )
            broadcast_shape[i] = rhs_shape[i] = max(a, b)

    lhs_type = VectorType.get(broadcast_shape, lhs_type.element_type)
    rhs_type = VectorType.get(broadcast_shape, rhs_type.element_type)
    if lhs_type != lhs.type:
        lhs = vector_d.broadcast(lhs_type, lhs)
    if rhs_type != rhs.type:
        rhs = vector_d.broadcast(rhs_type, rhs)
    return True, lhs, rhs


def _define_arithmetic_handlers():
    def register(py_operator, mnemonic):
        @handle_op(py_operator)
        def _(emitter: ThreadEmitter, node: fx.Node):
            try:
                lhs, rhs = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguments") from e

            lhs = cast_py_value(emitter, lhs)
            rhs = cast_py_value(emitter, rhs)
            is_vector, lhs, rhs = binary_broadcast(lhs, rhs)
            if is_vector:
                result = ScalarBuilder.binary_vector_arithmetic(mnemonic, lhs, rhs)
            else:
                result = ScalarBuilder.binary_arithmetic(mnemonic, lhs, rhs)
            emitter.bind_node_result(node, result)

    for py_operator, mnemonic in BINARY_ARITHMETIC_OPS:
        # Need to capture these per iteration, not just final value,
        # so call a function.
        register(py_operator, mnemonic)


_define_arithmetic_handlers()

###############################################################################
# Core data movement and indexing ops
###############################################################################


@handle_op(ops.thread_program_id)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        (axis,) = node.args
        axis = Index(axis)
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    value = emitter.lookup_grid_axis_value(axis)
    emitter.bind_node_result(node, value)


@handle_op(ops.kernel_buffer_getitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, slice_spec = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    sa = SliceAnalysis(kb_py_type.symbolic_shape, slice_spec)
    sa.normalize_symbolic_ranges()
    vector_shape = sa.symbolic_shape
    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    pad_attr = ScalarBuilder.zero_attr(element_type)
    indices = cast_indices(emitter, [s.start for s in sa.slices])
    pad_value = arith_d.constant(pad_attr)
    result = vector_d.transfer_read(
        vector_type,
        kb_src,
        indices,
        AffineMap.get_identity(len(indices)),
        pad_value,
    )
    emitter.bind_node_result(node, result)


@handle_op(ops.kernel_buffer_setitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, slice_spec, item = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    dest_rank = kb_ir_type.rank
    sa = SliceAnalysis(kb_py_type.symbolic_shape, slice_spec)
    sa.normalize_symbolic_ranges()
    indices = cast_indices(emitter, [s.start for s in sa.slices])
    if dest_rank != len(indices):
        raise CodegenError(
            f"Mismatched slice assignment: Expected rank {dest_rank}, got {len(indices)}"
        )
    insert_vector = cast_vector(emitter, item, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    insert_rank = insert_type.rank

    # Special case rank-0 broadcast.
    if insert_rank == 0:
        broadcast_type = VectorType.get(dest_rank * [1], kb_ir_type.element_type)
        insert_vector = vector_d.broadcast(broadcast_type, insert_vector)

    permutation_map = AffineMap.get_identity(dest_rank)
    vector_d.transfer_write(
        None,
        insert_vector,
        kb_dest,
        indices,
        AffineMapAttr.get(permutation_map),
    )


###############################################################################
# Memory Ops
###############################################################################


@handle_op(tkl.load)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, multi_index, vector_shape = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    pad_attr = ScalarBuilder.zero_attr(element_type)
    indices = cast_indices(emitter, multi_index)
    pad_value = arith_d.constant(pad_attr)
    result = vector_d.transfer_read(
        vector_type,
        kb_src,
        indices,
        AffineMap.get_identity(len(indices)),
        pad_value,
    )
    emitter.bind_node_result(node, result)


@handle_op(tkl.store)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, multi_index, item = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, kb)
    dest_rank = kb_ir_type.rank
    indices = cast_indices(emitter, multi_index)
    if dest_rank != len(indices):
        raise CodegenError(
            f"Mismatched slice assignment: Expected rank {dest_rank}, got {len(indices)}"
        )
    insert_vector = cast_vector(emitter, item, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    insert_rank = insert_type.rank

    # Special case rank-0 broadcast.
    if insert_rank == 0:
        broadcast_type = VectorType.get(dest_rank * [1], kb_ir_type.element_type)
        insert_vector = vector_d.broadcast(broadcast_type, insert_vector)

    permutation_map = AffineMap.get_identity(dest_rank)
    vector_d.transfer_write(
        None,
        insert_vector,
        kb_dest,
        indices,
        AffineMapAttr.get(permutation_map),
    )


###############################################################################
# Math Ops
###############################################################################


@handle_op(tkl.constant)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        shape, dtype, value = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # TODO: Have better way to get the dtype.
    if dtype == torch.float32:
        element_type = F32Type.get()
        vector_type = VectorType.get(shape, element_type)
        dense_value = DenseElementsAttr.get_splat(vector_type, FloatAttr.get_f32(value))
        result = arith_d.ConstantOp(vector_type, dense_value)
        emitter.bind_node_result(node, result)
    else:
        raise CodegenError(f"NYI: Constant type {dtype}")


###############################################################################
# Reduction Ops
###############################################################################


@handle_op(tkl.dot)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        lhs, rhs, acc = node.args
        lhs = cast_vector(emitter, lhs)
        rhs = cast_vector(emitter, rhs)
        acc = cast_vector(emitter, acc)
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_type = VectorType(lhs.type)
    element_type = vector_type.element_type
    rank = vector_type.rank

    n, m, k = (
        AffineExpr.get_dim(0),
        AffineExpr.get_dim(1),
        AffineExpr.get_dim(2),
    )
    indexing_maps = [
        AffineMap.get(3, 0, [n, k]),
        AffineMap.get(3, 0, [k, m]),
        AffineMap.get(3, 0, [n, m]),
    ]
    indexing_maps_attr = [AffineMapAttr.get(map) for map in indexing_maps]
    # TODO: Bad hack, please fix.
    iterator_types = ArrayAttr.get(
        [
            Attribute.parse("#vector.iterator_type<parallel>"),
            Attribute.parse("#vector.iterator_type<parallel>"),
            Attribute.parse("#vector.iterator_type<reduction>"),
        ]
    )
    result = vector_d.ContractionOp(
        acc.type,
        lhs,
        rhs,
        acc,
        indexing_maps_attr,
        iterator_types,
    )
    emitter.bind_node_result(node, result)


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(tkl.for_loop)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        start, end, step, init_args = node.args
        subgraph = node.kwargs["subgraph"]
        implicit_capture = node.kwargs["implicit_capture"]
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Check if init_args is a flattened list of values.
    for arg in init_args:
        if len(emitter.lookup_node_values(arg)) != 1:
            raise CodegenError(f"NYI: For loop init args must be flattened")

    # Get IR values mapping to the node args.
    start = cast_py_value(emitter, start)
    end = cast_py_value(emitter, end)
    step = cast_py_value(emitter, step)

    # Flatten init_args and get IR values for each of them.
    flat_init_args, init_args_spec = pytree.tree_flatten((init_args))
    flat_init_args = [cast_py_value(emitter, arg) for arg in flat_init_args]

    # Get the subgraph for body of the loop.
    assert isinstance(subgraph, str)
    subgraph = emitter.trace.get_subgraph(subgraph)

    # Create scf.for operation.
    forOp = scf_d.ForOp(start, end, step, flat_init_args)
    # Enter body of for loop.
    with InsertionPoint(forOp.body):
        # TODO: Flatten subgraph args here.
        subgraph_args = [
            node
            for node in subgraph.nodes
            if node.op == "placeholder" and "lifted" not in node.meta
        ]
        # Add mapping for induction variable argument.
        emitter.bind_node_result(subgraph_args[0], forOp.induction_variable)
        # Add mapping for iter_args.
        emitter.bind_node_results(subgraph_args[1], forOp.inner_iter_args)

        ret = emitter.emit_subgraph(subgraph, implicit_capture)
        # Use ret in terminatory of body
        # TODO: Flatten return values here.
        flat_ret_values, ret_spec = pytree.tree_flatten((ret))
        flat_ret_values = [cast_py_value(emitter, value) for value in flat_ret_values]
        scf_d.YieldOp(flat_ret_values)

    results = forOp.results_
    emitter.bind_node_results(node, results)


###############################################################################
# Torch and math ops
###############################################################################


@handle_op(torch.exp)
def _(emitter: ThreadEmitter, node: fx.Node):
    args = op_matchers.torch_exp(*node.args, **node.kwargs)
    raw_input = args["input"]
    input = cast_vector(emitter, raw_input)
    result = math_d.exp(input)
    emitter.bind_node_result(node, result)


@handle_op(torch.max)
def _(emitter: ThreadEmitter, node: fx.Node):
    args = op_matchers.torch_max_unary(
        *node.args, **node.kwargs
    ) or op_matchers.torch_max(*node.args, **node.kwargs)

    def combiner(element_type: IrType, attrs: NodeAttrs) -> vector_d.CombiningKind:
        if ScalarBuilder.is_floating_point_type(element_type):
            # Non-NaN propagating.
            # TODO: Carry a "fastmath" flag on the emitter and choose between this
            # and MAXIMUMF?
            return vector_d.CombiningKind.MAXF
        elif ScalarBuilder.is_integer_type(element_type):
            return (
                vector_d.CombiningKind.MAXUI
                if attrs.unsigned
                else vector_d.CombiningKind.MAXSI
            )

    emit_reduction(emitter, node, args, combiner)


@handle_op(torch.sum)
def _(emitter: ThreadEmitter, node: fx.Node):
    args = op_matchers.torch_sum_unary(
        *node.args, **node.kwargs
    ) or op_matchers.torch_sum(*node.args, **node.kwargs)

    def combiner(element_type: IrType, attrs: NodeAttrs) -> vector_d.CombiningKind:
        return vector_d.CombiningKind.ADD

    emit_reduction(emitter, node, args, combiner)


def emit_reduction(
    emitter: ThreadEmitter,
    node: fx.Node,
    args: dict,
    combiner_callback: Callable[[IrType], vector_d.CombiningKind],
):
    # Setup.
    raw_input = args["input"]
    attrs = NodeAttrs.load(raw_input)
    input = cast_vector(emitter, raw_input)
    vector_type = VectorType(input.type)
    element_type = vector_type.element_type
    rank = vector_type.rank
    zero = arith_d.constant(ScalarBuilder.zero_attr(element_type))
    combiner = combiner_callback(element_type, attrs)

    if len(args) == 1:
        # Reduce to scalar.
        scalar_result = vector_d.multi_reduction(
            combiner, input, zero, list(range(rank))
        )
        result = vector_d.splat(VectorType.get([], element_type), scalar_result)
        emitter.bind_node_result(node, result, attrs=attrs)
    else:
        # Reduce to vector.
        raise CodegenError("NYI: Reduce to vector")


###############################################################################
# Conversion utilities
###############################################################################


def cast_py_value(emitter: ThreadEmitter, value) -> Value:
    """
    Converts the given value to an IR Value.
    If the value is a fx.Node, the result of the fx.Node should corresspond to
    exactly one IR Value.
    If the value is a constant, a constant value will be built for it.
    """
    if isinstance(value, fx.Node):
        try:
            node_values = emitter.lookup_node_values(value)
            assert len(node_values) == 1, f"Expected exactly one value for node {value}"
            return node_values[0]
        except KeyError:
            raise CodegenError(f"Producer node `{value}` has no IR Value")

    return ScalarBuilder.constant(value)


def cast_py_lvalue(emitter: ThreadEmitter, py_value: fx.Node) -> tuple[Value, fx.Node]:
    if isinstance(py_value, fx.Node):
        try:
            node_values = emitter.lookup_node_values(py_value)
            assert (
                len(node_values) == 1
            ), f"Expected exactly one value for node {py_value}"
            return node_values[0], py_value
        except KeyError:
            raise CodegenError(f"Producer node `{py_value}` has no IR Value")
    else:
        raise CodegenError(
            f"Required a traced node in the graph. Got: {py_value} (type {type(py_value)})"
        )


def cast_kernel_buffer(
    emitter: ThreadEmitter, kb
) -> tuple[Value, MemRefType, Type[KernelBuffer]]:
    """Casts a Python value of type KernelBuffer, which lowers to a MemRefType'd value."""
    value, node = cast_py_lvalue(emitter, kb)
    ir_type = value.type
    py_type = node.type

    if not MemRefType.isinstance(ir_type):
        raise CodegenError(
            f"Expected a KernelBuffer (aka. `memref`) but got `{ir_type}`"
        )

    if not issubclass(py_type, KernelBuffer):
        raise CodegenError(
            f"Expected an lvalue of type KernelBuffer but got '{py_type}' for node {node}"
        )

    return value, MemRefType(ir_type), py_type


def cast_indices(emitter: ThreadEmitter, slice) -> list[Value]:
    if not isinstance(slice, (tuple, list)):
        slice = (slice,)
    ir_slice = [cast_py_value(emitter, s) for s in slice]
    return ir_slice


def cast_vector(
    emitter: ThreadEmitter, value, *, element_type: Optional[IrType] = None
):
    value = cast_py_value(emitter, value)

    # Promote scalar types correctly first.
    if element_type and not ShapedType.isinstance(value.type):
        # Implicit scalar type promotion.
        value = ScalarBuilder.promote(value, element_type)

    # After scalar promotion, promote to vector.
    if VectorType.isinstance(value.type):
        # Already a vector. Coerce or return.
        if element_type is not None:
            vector_type = VectorType(value.type)
            if vector_type.element_type == element_type:
                return value
            # TODO: Implement vector element type conversion.
            raise CodegenError(
                f"Implicit conversion of vector element types not supported (`{vector_type.element_type}` -> `{element_type}`)"
            )
        # No target element_type.
        return value
    else:
        # Scalar -> vector.
        element_type = value.type
        vector_type = VectorType.get([], element_type)
        return vector_d.splat(vector_type, value)

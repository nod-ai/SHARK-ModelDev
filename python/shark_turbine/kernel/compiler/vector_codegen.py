from typing import Any, Callable, Type, Optional, Sequence, Union

from dataclasses import dataclass
import operator as py_operator

import torch.fx as fx

from .._support.indexing import (
    Grid,
    IndexingContext,
    KernelBuffer,
    SymbolDef,
    _is_kernel_buffer_meta_derived,
)

from ..lang import (
    Index,
)

from .. import ops

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
    AffineMapAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IrType,
    Location,
    MemRefType,
    ShapedType,
    Value,
    VectorType,
    func_d,
    vector_d,
)


ArgTypeUnion = Union[SymbolDef, Type[KernelBuffer]]


@dataclass
class ArgMeta:
    name: Optional[str] = None
    node: Optional[fx.Node] = None
    grid_index: Optional[int] = None


class Signature:
    """Represents a function signature.

    Signatures can carry:
      - Input, output and temporary KernelBuffers
      - SymbolDef

    For now, if we enounter any of these, we emit them in declaration order.
    We need a better convention than this (i.e. inputs, then outputs, them symbols, them temporaries).
    """

    def __init__(self):
        self.args: list[tuple[ArgMeta, ArgTypeUnion]] = []

    def add_kernel_buffer(
        self, kb: Type[KernelBuffer], *, meta: Optional[ArgMeta] = None
    ):
        self.args.append((meta if meta is not None else ArgMeta(), kb))

    def add_symbol(self, sym: SymbolDef, *, meta: Optional[ArgMeta] = None):
        self.args.append((meta if meta is not None else ArgMeta(), sym))

    @property
    def arg_metas(self) -> Sequence[ArgMeta]:
        return (meta for meta, _ in self.args)

    def as_function_type(self) -> FunctionType:
        idx_c = IndexingContext.current()

        def sym_to_dim_asm(s: SymbolDef) -> str:
            static_value = idx_c.get_static_value(s)
            return "?" if static_value is None else str(static_value)

        def as_mlir_type(t: ArgTypeUnion) -> FunctionType:
            if isinstance(t, SymbolDef):
                return IndexType.get()
            elif _is_kernel_buffer_meta_derived(t):
                kb_t = t  # type: KernelBuffer
                element_type_asm = kb_t.element_type.ir_type_asm()
                symbolic_shape = kb_t.symbolic_shape
                if symbolic_shape is not None:
                    shape_asm = "x".join(sym_to_dim_asm(s) for s in kb_t.symbolic_shape)
                    spec_asm = f"{shape_asm}x{element_type_asm}"
                else:
                    # Unranked. Not well supported, but for completeness.
                    spec_asm = element_type_asm
                memref_asm = f"memref<{spec_asm}>"
                return IrType.parse(memref_asm)

        inputs = [as_mlir_type(arg) for _, arg in self.args]
        return FunctionType.get(inputs, [])

    def add_grid(self, grid: Type[Grid]):
        assert grid.symbolic_shape, "code emission requires a symbolically shaped grid"
        for index, s in enumerate(grid.symbolic_shape):
            self.add_symbol(s, meta=ArgMeta(grid_index=index, name=f"grid{index}"))

    def add_from_graph_placeholders(self, graph: fx.Graph):
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            t = node.type
            meta = ArgMeta(name=node.target, node=node)
            if _is_kernel_buffer_meta_derived(t):
                self.add_kernel_buffer(t, meta=meta)
            elif issubclass(t, SymbolDef):
                self.add_symbol(t, meta=meta)

    def __repr__(self):
        parts = []
        for meta, arg in self.args:
            part = repr(arg)
            if meta.name:
                part = f"{meta.name}: {part}"
            parts.append(part)
        return f"Signature({', '.join(parts)})"


class ThreadEmitter:
    """Emits a 'thread function' as a `func` with a signature derived from the gm."""

    OP_HANDLERS: dict[Any, Callable[["ThreadEmitter", fx.Node], None]] = {}

    def __init__(self, mb: ModuleBuilder, grid: Grid, sig: Signature):
        self.nv_map: dict[fx.Node, Value] = {}
        self.grid_index_map: list[Optional[Value]] = [None] * grid.rank

        # TODO: Infer a location from graph.
        with InsertionPoint(mb.body_block), Location.unknown():
            ftype = sig.as_function_type()
            func_op = func_d.FuncOp("kernel", ftype)
            self.func_op = func_op
            arg_locs = [
                (
                    Location.name(meta.name)
                    if meta.name is not None
                    else Location.unknown()
                )
                for meta in sig.arg_metas
            ]
            self.entry_block = func_op.add_entry_block(arg_locs)

            # Bind all inputs in the node-value map.
            for block_arg, meta in zip(self.entry_block.arguments, sig.arg_metas):
                assert (
                    meta.node or meta.grid_index is not None
                ), "expected all signature args to have an associated node or grid_index"
                if meta.node:
                    self.nv_map[meta.node] = block_arg
                elif meta.grid_index is not None:
                    self.grid_index_map[meta.grid_index] = block_arg
        self.context = mb.context
        self.ip = InsertionPoint(self.entry_block)

    def bind_node_result(
        self, node: fx.Node, value: Value, type_expr: Optional[type] = None
    ):
        assert node not in self.nv_map, f"Cannot rebind node {node}: already bound"
        if type_expr is not None:
            node.type = type_expr
        self.nv_map[node] = value

    def emit_graph(self, graph: fx.Graph):
        context = self.context
        for node in graph.nodes:
            # TODO: Construct a location for the node.
            with self.ip, Location.unknown(context):
                if node.op == "call_function":
                    self.emit_function_call_node(node)

    def emit_function_call_node(self, node: fx.Node):
        target_op = node.target
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")
        handler(self, node)

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

BINARY_ARITHMETIC_OPS = [
    (py_operator.add, "add"),
    (py_operator.mul, "mul"),
    (py_operator.sub, "sub"),
    (py_operator.mod, "mod"),
    (py_operator.floordiv, "floordiv"),
]


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

    try:
        value = emitter.grid_index_map[axis]
        assert value is not None, "Grid axis unbound"
    except IndexError as e:
        raise CodegenError("Grid axis out of bounds") from e

    emitter.bind_node_result(node, value, Index)


@handle_op(ops.kernel_buffer_getitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    raise CodegenError("NYI: kernel_buffer_getitem")


@handle_op(ops.kernel_buffer_setitem)
def _(emitter: ThreadEmitter, node: fx.Node):
    try:
        kb, key, item = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    kb_dest, kb_type = cast_kernel_buffer(emitter, kb)
    dest_rank = kb_type.rank
    indices = cast_indices(emitter, key)
    if dest_rank != len(indices):
        raise CodegenError(
            f"Mismatched slice assignment: Expected rank {dest_rank}, got {len(indices)}"
        )
    insert_vector = cast_vector(emitter, item, element_type=kb_type.element_type)
    insert_type = VectorType(insert_vector.type)
    insert_rank = insert_type.rank

    # This form only supports 0d broadcast or same rank currently.
    # TODO: This was specially crafted to make the iota demo work. Need to work
    # it out in generality.
    if insert_rank != 0 and insert_rank != dest_rank:
        raise CodegenError(
            f"The shorthand kernel_buffer[...]= assignment syntax only supports same rank assignment or restricted, 0d broadcast"
        )

    if insert_rank == 0:
        broadcast_type = VectorType.get(dest_rank * [1], kb_type.element_type)
        insert_vector = vector_d.broadcast(broadcast_type, insert_vector)
    permutation_map = AffineMap.get_identity(dest_rank)
    vector_d.transfer_write(
        None, insert_vector, kb_dest, indices, AffineMapAttr.get(permutation_map)
    )


###############################################################################
# Conversion utilities
###############################################################################


def cast_py_value(emitter: ThreadEmitter, value) -> Value:
    if isinstance(value, fx.Node):
        try:
            return emitter.nv_map[value]
        except KeyError:
            raise CodegenError(f"Producer node `{value}` has no IR Value")

    return ScalarBuilder.constant(value)


def cast_kernel_buffer(emitter: ThreadEmitter, kb) -> tuple[Value, MemRefType]:
    """Casts a Python value of type KernelBuffer, which lowers to a MemRefType'd value."""
    value = cast_py_value(emitter, kb)
    value_type = value.type

    if MemRefType.isinstance(value_type):
        return value, MemRefType(value_type)

    raise CodegenError(
        f"Expected a KernelBuffer (aka. `memref`) but got `{value_type}`"
    )


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
    if not ShapedType.isinstance(value.type):
        if element_type is not None:
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

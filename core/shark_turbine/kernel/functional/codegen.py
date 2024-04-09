from typing import Any, Callable, Type, Optional, Sequence, Union, List
from dataclasses import dataclass
import torch.fx as fx

from .._support.indexing import (
    IndexExpr,
    IndexingContext,
    IndexSymbol,
    SymIndex,
    index_expr,
)

from .._support.tracing import CapturedTrace
from .functional_ops import memory_to_register, mma
from ..compiler.builder import (
    IRProxyValue,
    ScalarBuilder,
)

from ..compiler.base import (
    CodegenError,
    NDEBUG,
    ValidationError,
)

from ..compiler.ir import (
    AffineMap,
    Attribute,
    AffineExpr,
    AffineMapAttr,
    ArrayAttr,
    FunctionType,
    VectorType,
    DenseElementsAttr,
    F32Type,
    IndexType,
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

from .. import lang as tkl

from ..compiler.kernel_codegen import (
    BoundKernelSignature,
)

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


class WaveEmitter:
    """Emits a 'warp function' as a `func` with a signature derived from the gm."""

    OP_HANDLERS: dict[Any, Callable[["WaveEmitter", fx.Node], None]] = {}

    def __init__(self, root_sig: BoundKernelSignature, trace: CapturedTrace):
        self._node_values: dict[fx.Node, List[IRProxyValue]] = {}
        self._root_sig = root_sig
        self.trace = trace
        self.ip = InsertionPoint(root_sig.entry_block)

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        assert NDEBUG or isinstance(node, fx.Node)
        values = self._node_values.get(node)
        if values is None:
            values = [self._root_sig.resolve_by_reference(("node", node))]
            self._node_values[node] = values
        return values

    def bind_node_proxy(
        self, node: fx.Node, proxy: IRProxyValue, *, attrs: Optional[NodeAttrs] = None
    ):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = [proxy]

    def bind_node_proxies(
        self,
        node: fx.Node,
        proxies: list[IRProxyValue],
        *,
        attrs: Optional[NodeAttrs] = None,
    ):
        """Binds a node's result to a list of Python/IR proxy object."""
        assert NDEBUG or (
            all(isinstance(p, IRProxyValue) for p in proxies)
            and isinstance(node, fx.Node)
        )
        assert (
            node not in self._node_values
        ), f"Cannot rebind node {node}: already bound"
        if attrs is not None:
            attrs.store(node)
        self._node_values[node] = proxies

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
    def decorator(f: Callable[["WaveEmitter", fx.Node], None]):
        WaveEmitter.OP_HANDLERS[op] = f
        return None

    return decorator

###############################################################################
# Python/scalar ops
###############################################################################

###############################################################################
# Core data movement and indexing ops
###############################################################################

###############################################################################
# Memory Ops
###############################################################################
@handle_op(memory_to_register)
def _(emitter: WaveEmitter, node: fx.Node):
    breakpoint()
    pass

###############################################################################
# Math Ops
###############################################################################
@handle_op(mma)
def _(emitter: WaveEmitter, node: fx.Node):
    breakpoint()
    pass

###############################################################################
# Control Flow ops
###############################################################################

###############################################################################
# Shape Manipulation Ops
###############################################################################

###############################################################################
# Conversion utilities
###############################################################################

###############################################################################
# Slice and indexing
###############################################################################
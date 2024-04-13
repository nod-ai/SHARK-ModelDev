from typing import (
    Type,
    Callable,
    Optional,
    Dict
)

import inspect
import math
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel as tk

import torch
import torch.fx as fx

from ..lang import (
    KernelBuffer,
    Grid,
    IndexExpr,
)

from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    EagerContext,
    Launchable,
    KernelRegionGraph,
    LaunchContext,
    AOTLaunchContext,
)

from .._support.indexing import IndexingContext

from ..compiler import (
    kernel_codegen,
    dispatch_codegen,
    builder,
    vector_codegen,
    host_codegen,
)

from ..compiler.ir import (
    Context,
    Operation,
)

from ..functional.codegen import WaveEmitter
from ..lang.functional_types import Register
from .constraints import ConstraintsMeta, WorkgroupConstraint, TilingConstraint, ThreadConstraint

__all__ = [
    "wave",
    "tiledLoop",
]


def wave(constraints: list[ConstraintsMeta]):

    def decorator(f: Callable) -> "LaunchableWave":
        return LaunchableWave(constraints, f.__name__, f)

    return decorator


def tiledLoop(*symbolic_dims: IndexExpr):
    # TODO: Use the argument to determine how many iterations
    def decorator(f: Callable):
        def wrapper(*args, **kwargs):
            # TODO: Here we need the maybe_scf_for()
            return f(args)
        return wrapper
    return decorator


class TiledLoop:
    def __init__(self, reduction_dims, name, function: Callable):
        self._name = name
        self._reduction_dims = reduction_dims
        self._f = function

    def __repr__(self):
        return f"tk.tiledLoop @{self._name}[{self._reduction_dims}]"


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: list[ConstraintsMeta],
        name: str,
        eager_function: Callable,
    ):
        super().__init__(eager_function)
        self.constraints = constraints
        self.grid_type = Grid[*self.get_grid_shape(constraints)]
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

    def get_grid_shape(self, constraints: list[ConstraintsMeta]) -> list[IndexExpr]:
        grid = [None, None]
        for constraint in constraints:
            if isinstance(constraint, WorkgroupConstraint):
                grid[constraint.workgroup_dim] = constraint.dim // constraint.tile_size
        return grid

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)
        return trace

    def eager_execute(self, args, kwargs):
        grid = self.grid_type()
        rank = grid.rank
        with EagerContext(rank=rank) as context:
            sig = self._sig
            bound = sig.bind(*args, *kwargs)
            bound.apply_defaults()
            # Transform args to KernelBuffers.
            for arg_name in list(bound.arguments.keys()):
                arg_value = bound.arguments[arg_name]
                param = sig.parameters[arg_name]
                param_type = param.annotation
                if isinstance(param_type, type) and issubclass(
                    param_type, KernelBuffer
                ):
                    kernel_buffer = param_type(arg_value)
                    bound.arguments[arg_name] = kernel_buffer
            volume = math.prod(grid)
            current_thread = context.current_thread
            for it in range(volume):
                for i in range(rank - 1):
                    current_thread[i] = it // grid[i]
                    it = it % grid[i]
                current_thread[-1] = it
                self._eager_function(*bound.args, **bound.kwargs)

    def propagate_types_in_graph(self, graph: fx.Graph, type_map: Dict[str, Type],
                                 subgraphs: Dict[str, fx.Node]):
        def look_for_type(node: fx.Node) -> Type:
            for input in node.all_input_nodes:
                if input.name in type_map:
                    return type_map[input.name]
            return None

        for node in graph.nodes:
            if node.op == 'placeholder':
                if node.name in type_map:
                    node.meta['type'] = type_map[node.name]
                    continue
                node.meta['type'] = type_map[node.name] = node.type
            if node.name == 'construct_register_from_metadata':
                args = [x for x in node.args[0]] + [node.args[1]]
                type_map[node.name] = node.meta['type'] = Register[*args]
            if 'write' in node.name or 'read' in node.name:
                arg_type = look_for_type(node)
                if arg_type is not None:
                    type_map[node.name] = node.meta['type'] = arg_type
            if 'subgraph' in node.kwargs:
                subgraph = subgraphs[node.kwargs['subgraph']]
                implicit_capture_nodes = []
                if 'implicit_capture' in node.kwargs:
                    implicit_capture_nodes += node.kwargs['implicit_capture']
                subgraph_inputs = list(set(node.all_input_nodes) - set(implicit_capture_nodes))
                i = 0
                for subnode in subgraph.nodes:
                    if 'type' not in subnode.meta:
                        subnode.meta['type'] = {}
                    if subnode.op == 'placeholder':
                        if subnode.name in type_map:
                            subnode.meta['type'] = type_map[subnode.name]
                            continue
                        subnode.meta['type'] = type_map[subnode.name] = type_map[subgraph_inputs[i].name]
                        i += 1
        return type_map

    """
    At the end of this function, all the placeholders in the graph
    should be annotated with a type accessible in the node's metadata
    node.meta['type']. Furthermore, we also annotate
    nodes that are registers constructed from metadata, and all
    the read and write nodes.
    """
    def propagate_types(self, trace: CapturedTrace):
        root_graph = trace.get_root_graph()
        subgraphs = trace.region_graph.subgraphs
        type_map = {}
        for graph in subgraphs.values():
            for node in graph.nodes:
                if 'type' not in node.meta:
                    node.meta['type'] = None
        type_map = self.propagate_types_in_graph(root_graph, type_map, subgraphs)
        for graph in subgraphs.values():
            if graph == root_graph:
                continue
            type_map = self.propagate_types_in_graph(graph, type_map, subgraphs)

    def propagate_constraints(self, trace: CapturedTrace):

        subgraphs = trace.region_graph.subgraphs
        root_graph = trace.get_root_graph()
        self.placeholders = {}
        self.induction_vars = {}
        i = 0
        for node in root_graph.nodes:
            if node.name == 'tiled_loop':
                self.induction_vars[node.args[0]] = tkl.IndexSymbol('ARG' + str(i))
                i += 1

        # Get thread constraints
        self.thread_constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, ThreadConstraint):
                self.thread_constraints.append(constraint)

        # Propagate constraints in root graph and subgraphs.
        for graph in subgraphs.values():
            for node in graph.nodes:
                if node.meta['type'] is not None:
                    shape = node.meta['type'].symbolic_shape
                    if 'index' not in node.meta:
                        node.meta['index'] = [0 for _ in range(len(shape))]
                    for idx, dim in enumerate(shape):
                        for constraint in self.constraints:
                            if isinstance(constraint, WorkgroupConstraint):
                                if dim == constraint.dim:
                                    node.meta['index'][idx] += constraint.apply()
                            if isinstance(constraint, TilingConstraint):
                                if dim == constraint.dim:
                                    node.meta['index'][idx] += constraint.apply(self.induction_vars[dim])
                if node.name == 'mma':
                    for i, arg in enumerate(node.args):
                        for constraint in self.thread_constraints:
                            matrix_type = None
                            match i:
                                case 0: matrix_type = 'A'
                                case 1: matrix_type = 'B'
                                case 2: matrix_type = 'C'
                            offset = constraint.apply(matrix_type)
                            for j in range(len(offset)):
                                arg.meta['index'][j] += offset[j]

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ):
        # Trace the function.
        trace = self._trace()
        idxc = IndexingContext.current()

        sig = self._sig
        bound = sig.bind(*args, *kwargs)
        bound.apply_defaults()

        for arg_name in list(bound.arguments.keys()):
            arg_value = bound.arguments[arg_name]
            param = sig.parameters[arg_name]
            param_type = param.annotation
            if isinstance(param_type, type) and issubclass(param_type, KernelBuffer):
                assert isinstance(arg_value, torch.Tensor)
                idxc.bind_shaped(arg_name, param_type, list(arg_value.shape))

        idxc.finalize()

        # Do type propagation to all nodes in subgraphs
        self.propagate_types(trace)
        # Propagate constraints to all nodes in the graph
        self.propagate_constraints(trace)

        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(trace.get_root_graph())
        kernel_sig.add_grid(self.grid_type)
        kernel_sig.determine_input_output_buffers(trace.get_root_graph())

        grid = self.grid_type()

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        dispatch_entrypoint = exe.define_entrypoint(entrypoint_name, kernel_sig, grid)
        emitter = WaveEmitter(dispatch_entrypoint, trace)
        emitter.emit()
        emitter.finish()

        mb.module_op.verify()

        return mb, exe, kernel_sig, entrypoint_name

    def test_execute(self, args, kwargs):
        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs
        )
        host_codegen.isolated_test_call(mb, exe, kernel_sig, entrypoint_name)

        print(mb.module_op.get_asm())

    def aot_execute(self, args, kwargs):
        launch_context = LaunchContext.current()
        assert isinstance(launch_context, AOTLaunchContext)

        module = launch_context.module

        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs, context=module.context, module_op=module.operation
        )

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"

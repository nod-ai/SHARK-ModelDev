from typing import Type, Callable, Optional, Dict

import inspect
import math
from functools import partial

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

from .._support.nodes import *

from ..compiler import (
    kernel_codegen,
    dispatch_codegen,
    builder,
    vector_codegen,
    host_codegen,
)

from ..compiler.ir import (
    builtin_d,
    Context,
    InsertionPoint,
    IrType,
    Location,
    Operation,
    gpu_d,
    transform_d,
    memref_d,
    UnitAttr,
    MemRefType,
    IntegerAttr,
    IndexType,
)

from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
    any_op_t,
)

from iree.compiler.dialects.transform.extras import apply_patterns, named_sequence
from iree.compiler.dialects import (
    _structured_transform_ops_gen as structured_transform_ops,
)

from ..functional.codegen import WaveEmitter, handle_read, handle_write
from ..functional.ops import alloc_shared, get_result, read_shared, write_shared
from ..functional import modulo_scheduling as ms

from ..lang.functional_types import Register, AddressSpace, Memory
from .constraints import (
    ConstraintsMeta,
    WorkgroupConstraint,
    TilingConstraint,
    ThreadConstraint,
    HardwareConstraint,
)

from ..compiler.builder import (
    IRProxyValue,
    ScalarBuilder,
)

from ..compiler.base import (
    CodegenError,
    NDEBUG,
    ValidationError,
)

from ..compiler.vector_codegen import (
    cast_py_literal,
    cast_py_value,
    cast_kernel_buffer,
    cast_slice_spec,
    cast_vector,
    extract_slice_starts,
)

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


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: list[ConstraintsMeta],
        name: str,
        eager_function: Callable,
    ):
        super().__init__(eager_function)
        self.constraints = constraints
        for hardware_constraint in self.hardware_constraints:
            for thread_constraint in self.thread_constraints:
                hardware_constraint.threads_per_block = (
                    thread_constraint.threads_per_block
                )

        self.grid_type = Grid[*self.get_grid_shape(constraints)]
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

    @property
    def workgroup_constraints(self):
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WorkgroupConstraint)
        ]

    @property
    def tiling_constraints(self):
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, TilingConstraint)
        ]

    @property
    def thread_constraints(self):
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, ThreadConstraint)
        ]

    @property
    def hardware_constraints(self):
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, HardwareConstraint)
        ]

    def get_grid_shape(self, constraints: list[ConstraintsMeta]) -> list[IndexExpr]:
        grid = [None, None]
        for constraint in self.workgroup_constraints:
            grid[constraint.workgroup_dim] = constraint.dim // constraint.tile_size
        return grid

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)
        return trace

    def canonicalize_module(self, module: Operation):
        with module.context, Location.unknown():

            transform_module = builtin_d.Module.create()
            transform_module_op = module.operation
            transform_module_op.attributes["transform.with_named_sequence"] = (
                UnitAttr.get()
            )
            with InsertionPoint(transform_module.body):
                named_seqence = transform_d.NamedSequenceOp(
                    "__transform_main", [any_op_t()], []
                )
                with InsertionPoint(named_seqence.body):
                    target = named_seqence.body.arguments[0]
                    # TODO: For now no canonicalization as that also removes
                    #       dead code. Currently in particular the workgroup_id
                    #       and thread_id operations.
                    # @apply_patterns(target)
                    # def patterns():

                    apply_patterns = transform_d.ApplyPatternsOp(target)
                    with InsertionPoint(apply_patterns.regions[0].blocks[0]):
                        transform_d.apply_patterns_canonicalization()

                    loops = structured_transform_ops.structured_match(
                        any_op_t(), target, ops=["scf.for"]
                    )
                    transform_d.apply_licm(loops)
                    transform_d.YieldOp([target])
            transform_interpreter.apply_named_sequence(
                module,
                transform_module.body.operations[0],
                transform_module,
            )

    def lower_module(self, module: Operation):
        with module.context, Location.unknown():

            transform_module = builtin_d.Module.create()
            transform_module_op = module.operation
            transform_module_op.attributes["transform.with_named_sequence"] = (
                UnitAttr.get()
            )
            with InsertionPoint(transform_module.body):
                named_seqence = transform_d.NamedSequenceOp(
                    "__transform_main", [any_op_t()], []
                )
                with InsertionPoint(named_seqence.body):
                    target = named_seqence.body.arguments[0]
                    target = transform_d.ApplyRegisteredPassOp(
                        any_op_t(), target, "convert-scf-to-cf"
                    )
                    # TODO: This one does not work as one of our functions does
                    #       not respect IREE conventions:
                    #       public functions on executables must be () -> ()

                    # transform_d.ApplyRegisteredPassOp(
                    #     any_op_t(), target, "iree-convert-to-llvm"
                    # )
                    transform_d.YieldOp([target])

            transform_interpreter.apply_named_sequence(
                module,
                transform_module.body.operations[0],
                transform_module,
            )

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def propagate_types_in_graph(
        self, graph: fx.Graph, type_map: Dict[str, Type], subgraphs: Dict[str, fx.Node]
    ):
        def look_for_type(node: fx.Node) -> Type:
            for input in node.all_input_nodes:
                if input.name in type_map:
                    return type_map[input.name]
            return None

        for node in graph.nodes:
            typed_node = getNode(node)
            if node.op == "placeholder":
                if node.name in type_map:
                    node.meta["type"] = type_map[node.name]
                    node.type = type_map[node.name]
                    continue
                node.meta["type"] = type_map[node.name] = node.type
            if node.name == "construct_register_from_metadata":
                args = [x for x in node.args[0]] + [node.args[1]]
                type_map[node.name] = node.meta["type"] = Register[*args]
            if "write" in node.name or "read" in node.name:
                arg_type = look_for_type(node)
                if arg_type is not None:
                    type_map[node.name] = node.meta["type"] = arg_type
            if isinstance(typed_node, TiledLoop):
                subgraph = subgraphs[typed_node.subgraph_name]
                implicit_capture_nodes = []
                # if "implicit_capture" in node.kwargs:
                implicit_capture_nodes += typed_node.implicit_captures
                subgraph_inputs = list(
                    set(node.all_input_nodes) - set(implicit_capture_nodes)
                )
                i = 0
                for subnode in subgraph.nodes:
                    if "type" not in subnode.meta:
                        subnode.meta["type"] = {}
                    if subnode.op == "placeholder":
                        if subnode.name in type_map:
                            subnode.meta["type"] = type_map[subnode.name]
                            continue
                        subnode.meta["type"] = type_map[subnode.name] = type_map[
                            subgraph_inputs[i].name
                        ]
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
                if "type" not in node.meta:
                    node.meta["type"] = None
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
            if node.name == "tiled_loop":
                self.induction_vars[node.args[0]] = tkl.IndexSymbol("ARG" + str(i))
                i += 1

        # Propagate constraints in root graph and subgraphs.
        for graph in subgraphs.values():
            for node in graph.nodes:
                if node.meta["type"] is not None:
                    shape = node.meta["type"].symbolic_shape
                    if "index" not in node.meta:
                        node.meta["index"] = [0 for _ in range(len(shape))]
                    for idx, dim in enumerate(shape):
                        for constraint in self.workgroup_constraints:
                            if dim == constraint.dim:
                                node.meta["index"][idx] += constraint.apply()
                        for constraint in self.tiling_constraints:
                            if dim == constraint.dim:
                                node.meta["index"][idx] += constraint.apply(
                                    self.induction_vars[dim]
                                )
                if node.name == "mma":
                    for i, arg in enumerate(node.args):
                        for constraint in self.hardware_constraints:
                            matrix_type = None
                            match i:
                                case 0:
                                    matrix_type = "A"
                                case 1:
                                    matrix_type = "B"
                                case 2:
                                    matrix_type = "C"
                            offset = constraint.apply(matrix_type)
                            for j in range(len(offset)):
                                arg.meta["index"][j] += offset[j]

    def get_string(self, node: fx.Node, i: int, nested_region: bool) -> str:
        prefix = " "
        nested_region_prefix = "b" if nested_region else ""

        def initialize(prefix: str, nested_region: bool):
            return prefix if not nested_region else prefix + prefix

        if node.op == "placeholder":
            if node.name in self.index_map:
                return self.get_string(node.next, i, nested_region)
            value_prefix = nested_region_prefix if nested_region else ""
            self.index_map[node.name] = value_prefix + f"{str(i)}"
            asm_str = ""
            if i == 0 and not nested_region:
                asm_str = "func.func @main("
                asm_str += (
                    f"%{i}: Memory<{node.type.symbolic_shape}, {node.type.dtype}>"
                )
                while node.next.op == "placeholder":
                    asm_str += ", "
                    node = node.next
                    i += 1
                    asm_str += (
                        f"%{i}: Memory<{node.type.symbolic_shape}, {node.type.dtype}>"
                    )
                    self.index_map[node.name] = value_prefix + f"{str(i)}"
                asm_str += ") {\n"
            return asm_str + self.get_string(node.next, i + 1, nested_region)

        asm_str = initialize(prefix, nested_region)

        typed_node = getNode(node)
        if not isinstance(typed_node, UnknownNode) and not isinstance(
            typed_node, TiledLoop
        ):
            self.index_map[node.name] = f"{nested_region_prefix}{i}"
            return (prefix if not nested_region else prefix + prefix) + (
                f"%{nested_region_prefix}{i} = {typed_node.custom_string(self.index_map)}"
                + self.get_string(node.next, i + 1, nested_region)
            )

        # TODO: Move the other ops below also to custom nodes
        if "tiled_loop" in node.name:
            if nested_region:
                j = self.parent_id
                self.index_map[node.name] = f"{j}"
                self.parent = None
                self.parent_id = None
                return self.get_string(node.next, j + 1, False)
            asm_str += f"%{i} = "
            args_str = ""
            for j, iter_arg in enumerate(node.args[1]):
                args_str += f"%{nested_region_prefix}{str(j)} = %{self.index_map[iter_arg.name]}, "
            asm_str += f"scf.for (K, iter_args = [{args_str}]) {{\n"
            first_node = list(self.subgraphs[typed_node.subgraph_name].nodes)[0]
            self.parent = node
            self.parent_id = i
            return asm_str + self.get_string(first_node, 0, True)
        if "output" in node.name:
            if self.parent is not None:
                asm_str += "scf.yield "
            else:
                asm_str += "return"
            for arg in node.args:
                if arg is not None:
                    asm_str += f"%{self.index_map[arg.name]}, "
            if self.parent is not None:
                asm_str += "\n" + initialize(prefix, False) + "}\n"
                return asm_str + self.get_string(self.parent, i + 1, nested_region)
            else:
                asm_str += "\n}\n"
            return asm_str

        return asm_str

    def print(self, trace: CapturedTrace):
        self.index_map = {}
        self.subgraphs = trace.region_graph.subgraphs
        self.parent = None
        self.parent_id = None
        root = list(trace.get_root_graph().nodes)[0]
        asm_str = self.get_string(root, 0, False)
        print(asm_str)

    """
    Promotes tkf.reads to reads from shared memory if the
    address space of the memory operand is shared memory.
    Introduces additional nodes in the fx graph for
    readign and writing from registers to shared memory,
    as well as a shared memory allocation.
    """

    def promote_to_shared_memory(self, trace: CapturedTrace, idxc: IndexingContext):
        subgraphs = trace.region_graph.subgraphs
        root_graph = trace.get_root_graph()
        # Add additional frozen subs for shared memory
        SMEM_SPACE = tkl.sym.SMEM_SPACE
        idxc.frozen_subs.append((SMEM_SPACE, AddressSpace.SHARED_MEMORY.value))
        for graph in subgraphs.values():
            for node in graph.nodes:
                # typed_node = getNode(node)
                # TODO: rewrite this to use the match..case syntax with typed_node
                if "read" in node.name and "shared" not in node.name:
                    for arg in node.all_input_nodes:
                        if arg.meta["type"] is not None:
                            shape = arg.meta["type"].symbolic_shape
                            dtype = arg.meta["type"].dtype
                            address_space = arg.meta["type"].address_space
                            for sym, val in idxc.frozen_subs:
                                if sym == address_space:
                                    address_space = val
                                    break
                            if address_space != AddressSpace.SHARED_MEMORY.value:
                                continue
                            for user in node.users.keys():
                                if "write_shared" in user.name:
                                    continue
                            # Create alloc node
                            current = root_graph._root
                            while current.next.op == "placeholder":
                                current = current.next
                            root_graph.inserting_after(current)
                            type = Memory[*list(shape) + [SMEM_SPACE, dtype]]
                            alloc_node = AllocSharedNode(
                                root_graph, alloc_shared, shape, dtype, type
                            )
                            alloc_node.emit()
                            alloc = alloc_node.fx_node
                            alloc.meta["type"] = type

                            # Create read shared node
                            read_shared_node = ReadSharedNode(
                                graph,
                                read_shared,
                                alloc,
                                node.args[1],
                                type=Register[*list(shape) + [dtype]],
                            )

                            read_shared_node.emit()
                            read_shared_fx = read_shared_node.fx_node
                            read_shared_fx.meta["type"] = read_shared_node.type
                            node.replace_all_uses_with(read_shared_fx)

                            # Create write shared node
                            write_shared_node = WriteSharedNode(
                                graph, write_shared, node, alloc, node.args[1]
                            )
                            write_shared_node.emit()
                            write_shared_fx = write_shared_node.fx_node
                            write_shared_fx.meta["type"] = None
                            node.append(write_shared_fx)
                            write_shared_fx.append(read_shared_fx)

    """
    This function returns the resource reservation table (RRT) for
    different types of nodes. We assume the target machine
    has G global read/write units, S shared read/write units
    and M mfma units.
    ---------------------------------------------------
    |  Global Read/Write |  Shared Read/Write |  MMA  |
    ---------------------------------------------------
    |        G           |         S          |   M   |
    ---------------------------------------------------
    Every op in the graph can specify how many units it
    utilizes and for how long. For example, say we have
    a complex op that utilizes 2 units of the global memory subsytem
    and then uses 2 units of mma after, it would be represented
    as below.
    ---------------------------------------------------
    |  Global Read/Write |  Shared Read/Write |  MMA  |
    ---------------------------------------------------
    |        2           |         0          |   0   |
    ---------------------------------------------------
    |        0           |         0          |   2   |
    ---------------------------------------------------
    This function returns the RRT for a given op.
    """

    def get_rrt(self, name: str):
        if "read" in name or "write" in name:
            if "shared" not in name:
                return [[1, 0, 0]]
            else:
                return [[0, 1, 0]]
        if "mma" in name:
            return [[0, 0, 1]]
        return [[0, 0, 0]]

    """
    This function returns the delay of a given op in cycles.
    """

    def get_delay(self, name: str):
        if "read" in name or "write" in name:
            if "shared" not in name:
                return 5
            else:
                return 1
        if "mma" in name:
            return 2
        return 0

    def create_loop_graph(self, value_map, iter_args):
        kernel = fx.Graph()
        for stage, nodes in self.nodes_by_stage.items():
            for node in nodes:
                if "alloc" in node.name:
                    continue
                if "c_reg" in node.name:
                    continue
                node.meta["stage"] = stage - 1
                new_node = kernel.node_copy(node, lambda node: value_map[node.name])
                # This mapping is constantly updated to only the last reference.
                # Is that always enough?
                value_map[node.name] = new_node
        mapped_iter_args = []
        for arg in iter_args:
            if "c_reg" in arg.name:
                indices = arg.name.split("_")[-2:]
                new_name = "_".join(["mma"] + indices + ["1"])
                mapped_iter_args.append(value_map[new_name])
            else:
                mapped_iter_args.append(value_map[arg.name])
        kernel.create_node("output", "output", (mapped_iter_args,))
        return kernel

    def create_scheduled_graph(self, expanded_graph, scheduler, trace):
        """
        This pass uses the macrokernel graph and the generated schedule
        and constructs a scheduled macrokernel graph that can be used
        for MLIR emission.
        """
        root_graph = trace.get_root_graph()
        scheduled_graph = fx.Graph()

        value_map = {}
        c_reg_node = None
        new_tiled_loop = None

        def arg_mapper(node: fx.Node):
            if node.name in value_map:
                return value_map[node.name]
            if "c_reg" not in node.name:
                return node
            else:
                return c_reg_node

        def epilogue_arg_mapper(node: fx.Node):
            if node.name in value_map:
                return value_map[node.name]
            return node

        for node in root_graph.nodes:
            typed_node = getNode(node)
            if isinstance(typed_node, TiledLoop):
                # Emit prologue
                new_iter_args = []
                old_iter_args = []
                for stage in reversed(self.prologue.keys()):
                    for subnode in self.prologue[stage]:
                        if "c_reg" in subnode.name:
                            value_map[subnode.name] = c_reg_node
                            continue
                        subnode.meta["stage"] = stage - 1
                        new_node = scheduled_graph.node_copy(subnode, arg_mapper)
                        new_node.name = subnode.name + "_prolog" + str(stage - 1)
                        value_map[subnode.name] = new_node

                for stage, nodes in self.iter_args.items():
                    for subnode in nodes:
                        new_iter_args.append(value_map[subnode.name])
                        old_iter_args.append(subnode)

                for subnode in self.mma_args:
                    if subnode.name not in value_map:
                        value_map[subnode.name] = c_reg_node
                    new_iter_args.append(value_map[subnode.name])
                    old_iter_args.append(subnode)

                # Emit loop
                new_tiled_loop = scheduled_graph.node_copy(node)
                new_tiled_loop.args = (
                    node.args[0],
                    new_iter_args,
                    node.args[2],
                    [value_map[x.name] for x in node.args[3]],
                )
                loop_body = self.create_loop_graph(value_map, old_iter_args)
                new_tiled_loop.kwargs = {
                    "subgraph": loop_body,
                    "iter_args": old_iter_args,
                }
                value_map[node.name] = new_tiled_loop

                # Emit nodes representing indexing into the list of results of the loop
                for idx, arg in enumerate(old_iter_args):
                    get_res = GetResultNode(
                        scheduled_graph, get_result, new_tiled_loop, idx
                    )
                    get_res.emit()
                    value_map[arg.name] = get_res.fx_node

                # Emit epilogue
                for stage in reversed(self.epilogue.keys()):
                    for subnode in self.epilogue[stage]:
                        if "c_reg" in subnode.name:
                            continue
                        subnode.meta["stage"] = stage
                        new_node = scheduled_graph.node_copy(
                            subnode, epilogue_arg_mapper
                        )
                        new_node.name = subnode.name + "_epilog" + str(stage)
                        value_map[subnode.name] = new_node
                        # Whenever we compute the last mma node in an mma-chain,
                        # we need to update the value mapper for the corresponding
                        # c_reg.
                        if "mma" in subnode.name:
                            i, j, k = subnode.name.split("_")[-3:]
                            if int(k) == self.batch_k - 1:
                                c_reg_name = "_".join(["c_reg", i, j])
                                value_map[c_reg_name] = new_node
                continue
            new_node = scheduled_graph.node_copy(
                node, lambda node: value_map[node.name]
            )
            value_map[node.name] = new_node
            if "construct_register_from_metadata" in node.name:
                c_reg_node = new_node

        return scheduled_graph

    def create_expanded_graph(self, trace, idxc, debug=False):
        """
        This pass creates a new fx.Graph that corresponds to the "macrokernel".
        We use the "microkernel" graph (that we obtained from tracing the
        tkf program) and expand it to the "macrokernel" based on the user
        specified constraints.
        """

        # Determine how many nodes there are in the final graph.
        hardware_constraint = self.hardware_constraints[0]
        mma_m, mma_n, mma_k = hardware_constraint.mma_matrix_shapes()
        for sym, val in idxc.frozen_subs:
            if sym.name == "BLOCK_M":
                block_m = val
            if sym.name == "BLOCK_N":
                block_n = val
            if sym.name == "BLOCK_K":
                block_k = val
        self.batch_m = 2  # block_m // mma_m
        self.batch_n = 2  # block_n // mma_n
        self.batch_k = 2  # block_k // mma_k
        repeat_times = {"M": self.batch_m, "N": self.batch_n, "K": self.batch_k}

        subgraphs = trace.region_graph.subgraphs
        root_graph = trace.get_root_graph()
        expanded_graph = fx.Graph()

        index_suffix = lambda i, j: "_" + str(i) + "_" + str(j)
        mma_index_suffix = lambda i, j, k: index_suffix(i, j) + "_" + str(k)

        def transform_args(i: int, j: int, arg: fx.Node):
            if arg.op == "placeholder" or "alloc" in arg.name:
                return arg
            new_arg_name = arg.name + index_suffix(i, j)
            for node in expanded_graph.nodes:
                if node.name == new_arg_name:
                    return node
            return None

        def transform_mma_args(i: int, j: int, k: int, arg: fx.Node):
            if arg.op == "placeholder":
                if k == 0:
                    new_arg_name = arg.name + index_suffix(i, j)
                else:
                    new_arg_name = "mma" + mma_index_suffix(i, j, k - 1)
                for node in expanded_graph.nodes:
                    if node.name == new_arg_name:
                        return node
                return None

            # Determine if this is the lhs or rhs of the mma operation
            for user in arg.users.keys():
                if user.name == "mma":
                    for c, user_arg in enumerate(user.args):
                        if user_arg == arg:
                            index = c
                            break
            match index:
                case 0:
                    new_arg_name = arg.name + index_suffix(i, k)
                case 1:
                    new_arg_name = arg.name + index_suffix(j, k)
                case _:
                    return None
            for node in expanded_graph.nodes:
                if node.name == new_arg_name:
                    return node
            return None

        def duplicate_node(m: int, k: int, node: fx.Node):
            for i in range(m):
                for j in range(k):
                    new_node = expanded_graph.node_copy(
                        node, partial(transform_args, i, j)
                    )
                    new_node.name = node.name + index_suffix(i, j)

        def duplicate_mma_node(M: int, N: int, K: int, node: fx.Node):
            outputs = []
            for i in range(M):
                for j in range(N):
                    for k in range(K):
                        new_node = expanded_graph.node_copy(
                            node, partial(transform_mma_args, i, j, k)
                        )
                        new_node.name = node.name + mma_index_suffix(i, j, k)
                    outputs.append(new_node)
            return outputs

        for graph in subgraphs.values():
            if graph == root_graph:
                continue
            outputs = None
            for node in graph.nodes:
                if node.op == "placeholder" and "c_reg" not in node.name:
                    continue
                if "mma" in node.name:
                    outputs = duplicate_mma_node(
                        self.batch_m, self.batch_n, self.batch_k, node
                    )
                    continue
                if "output" in node.name:
                    new_node = expanded_graph.node_copy(node)
                    new_node.args = tuple(
                        [outputs],
                    )
                    continue
                node_type = node.meta["type"]
                if node_type is None:
                    # If type not available, must be a write, so get it
                    # from args.
                    for arg in node.args:
                        if arg.meta["type"] is not None:
                            node_type = arg.meta["type"]
                            break
                dim0, dim1 = [x.name for x in node_type.symbolic_shape]
                duplicate_node(repeat_times[dim0], repeat_times[dim1], node)
        return expanded_graph

    def construct_schedule(self, graph, debug=True):
        self.dependenceGraph = ms.Graph()
        self.node_mapper = {}
        self.inverse_mapper = {}
        for node in graph.nodes:
            # No need to add output, since we have c_reg
            if "output" in node.name:
                continue
            self.node_mapper[node] = ms.Node(node.name, self.get_rrt(node.name))
            self.inverse_mapper[self.node_mapper[node]] = node
            self.dependenceGraph.addNode(self.node_mapper[node])

        def get_output_to_node(node: fx.Node):
            _, i, j, _ = node.name.split("_")
            to_node_name = f"c_reg_{i}_{j}"
            for node in graph.nodes:
                if node.name == to_node_name:
                    return node
            return None

        for fromNode in graph.nodes:
            for toNode in fromNode.users.keys():
                iteration_delay = 0
                if "c_reg" in fromNode.name:
                    iteration_delay = 1
                if "output" in toNode.name:
                    toNode = get_output_to_node(fromNode)
                edge_label = f"{fromNode.name} -> {toNode.name}"
                self.dependenceGraph.addEdge(
                    ms.Edge(
                        edge_label,
                        self.node_mapper[fromNode],
                        self.node_mapper[toNode],
                        self.get_delay(fromNode.name),
                        iteration_delay,
                    )
                )

        # Create edges between write and read from shared memory.
        for node in graph.nodes:
            if "write_shared" in node.name:
                i, j = node.name.split("_")[-2:]
                prefix = "_".join(node.name.split("_")[:-2]).replace("write", "read")
                read_shared_name = prefix + "_" + str(i) + "_" + str(j)
                fromNode = node
                for nodej in graph.nodes:
                    if nodej.name == read_shared_name:
                        toNode = nodej
                        break
                self.dependenceGraph.addEdge(
                    ms.Edge(
                        edge_label,
                        self.node_mapper[fromNode],
                        self.node_mapper[toNode],
                        self.get_delay(fromNode.name),
                        0,
                    )
                )

        self.dependenceGraph.generateDotGraph()
        resourceVector = [2, 2, 2]
        scheduler = ms.ModuloScheduler(resourceVector, self.dependenceGraph)
        scheduler.generateSchedule()
        return scheduler

    def construct_prologue_and_epilogue(self, scheduler):
        """
        This function constructs the prologue and epilogue for a given schedule.
        The given schedule is broken into N stages. All stages > 0 have a contribution
        to the prologue. So the prologue for stage i, will be all the instructions in
        stages [0,i-1].

        Similarly, for the epilogue, all stages < N - 1 have a contribution to the
        epilogue. The epilogue for stage i, will be all the instructions from
        stages [i+1, N-1].

        We also keep track of the iter args that will be need to construct the for loop
        during mlir emission. The iter_args returned at the end of stage i, will be
        inputs for stage i+1. We use this criteria to determine the iter args. In
        addition, we also need to return the results of the final mmas.
        """

        # Ignore input placeholders, allocations and outputs
        def criteria(node: fx.Node):
            input_nodes = ["a", "b"]
            if node.name in input_nodes and node.op == "placeholder":
                return False
            ignore_nodes = ["alloc"]
            if any([x in node.name for x in ignore_nodes]):
                return False
            return True

        # Determine the init_args of the loop.
        initiation_iterval = len(scheduler.RT)
        sorted_schedule = dict(sorted(scheduler.schedule.items(), key=lambda x: x[1]))

        # Partition graph by stage and go back to using the fx.Nodes.
        self.nodes_by_stage = {}
        for node, t in sorted_schedule.items():
            stage = t // initiation_iterval
            if stage not in self.nodes_by_stage:
                self.nodes_by_stage[stage] = []
            inverse_node = self.inverse_mapper[node]
            if criteria(inverse_node):
                self.nodes_by_stage[stage].append(inverse_node)
            max_stage = stage

        self.prologue = {stage: [] for stage in range(max_stage + 1)}
        self.epilogue = {stage: [] for stage in range(max_stage + 1)}
        for stage in self.prologue.keys():
            for i in range(0, stage):
                self.prologue[stage] += self.nodes_by_stage[i]
        for stage in self.epilogue.keys():
            for i in range(stage + 1, max_stage + 1):
                self.epilogue[stage] += self.nodes_by_stage[i]

        self.iter_args = {stage: set() for stage in range(max_stage + 1)}
        self.mma_args = []
        for stage in self.nodes_by_stage.keys():
            for node in self.nodes_by_stage[stage]:
                if "c_reg" in node.name:
                    self.mma_args.append(node)
                    continue
                if stage == max_stage:
                    continue
                for user in list(node.users.keys()):
                    if (
                        user not in self.nodes_by_stage[stage]
                        and user in self.nodes_by_stage[stage + 1]
                    ):
                        self.iter_args[stage].add(node)

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

        # Do shared memory promotion if required
        self.promote_to_shared_memory(trace, idxc)

        # Propagate constraints to all nodes in the graph
        self.propagate_constraints(trace)

        # Create "macrokernel" graph
        expanded_graph = self.create_expanded_graph(trace, idxc)
        # Schedule "macrokernel" graph
        scheduler = self.construct_schedule(expanded_graph)
        # Construct prologue and epilogue
        self.construct_prologue_and_epilogue(scheduler)
        scheduled_graph = self.create_scheduled_graph(expanded_graph, scheduler, trace)

        # MLIR-style debug print of the graph
        self.print(trace)

        kernel_sig = kernel_codegen.KernelSignature()

        def ignore_criteria(node: fx.Node):
            return "c_reg" in node.name

        kernel_sig.add_from_graph_placeholders(scheduled_graph, ignore_criteria)
        kernel_sig.add_grid(self.grid_type)
        kernel_sig.determine_input_output_buffers(scheduled_graph)

        grid = self.grid_type()

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        dispatch_entrypoint = exe.define_entrypoint(entrypoint_name, kernel_sig, grid)
        emitter = WaveEmitter(dispatch_entrypoint, trace)

        emitter.emit(graph=scheduled_graph)
        emitter.finish()

        # self.canonicalize_module(mb.module_op)
        # self.lower_module(mb.module_op)
        mb.module_op.verify()

        return mb, exe, kernel_sig, entrypoint_name

    def test_execute(self, args, kwargs):
        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs
        )
        host_codegen.isolated_test_call(mb, exe, kernel_sig, entrypoint_name)

        print(mb.module_op.get_asm())

    def aot_execute(self, args, kwargs):
        assert isinstance(launch_context, AOTLaunchContext)

        module = launch_context.module

        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs, context=module.context, module_op=module.operation
        )

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"

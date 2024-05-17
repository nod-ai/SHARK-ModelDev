from typing import Type, Callable, Optional, Dict
import inspect
import math
from functools import partial
import sympy
import difflib
from copy import deepcopy

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
from ..functional.ops import (
    alloc_shared,
    barrier,
    get_result,
    read_shared,
    read,
    write_shared,
    sync,
)
from ..functional import modulo_scheduling as ms
from ..functional import utils

from ..lang.functional_types import Register, AddressSpace, Memory
from .constraints import (
    ConstraintsMeta,
    WorkgroupConstraint,
    TilingConstraint,
    WaveConstraint,
    HardwareConstraint,
    SchedulingConstraint,
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
        waves_per_block = [1, 1, 1]
        self.utils = utils.Utils(
            self.constraints[0].workgroup_ids, self.constraints[0].induction_variables
        )

        for workgroup_constraint in self.workgroup_constraints:
            wg_dim = workgroup_constraint.workgroup_dim
            for wave_constraint in self.wave_constraints:
                if wg_dim == wave_constraint.thread_dim:
                    waves_per_block[wg_dim] = (
                        workgroup_constraint.tile_size / wave_constraint.tile_size
                    )

        for hardware_constraint in self.hardware_constraints:
            hardware_constraint.waves_per_block = waves_per_block

        # Set defaults for delays
        self.global_delay = 5
        self.shared_delay = 1
        self.mma_delay = 2
        for scheduling_constraint in self.scheduling_constraints:
            for unit, delay in scheduling_constraint.delays.items():
                match unit:
                    case "GLOBAL":
                        self.global_delay = delay
                    case "SHARED":
                        self.shared_delay = delay
                    case "MMA":
                        self.mma_delay = delay

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
    def wave_constraints(self):
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WaveConstraint)
        ]

    @property
    def hardware_constraints(self):
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, HardwareConstraint)
        ]

    @property
    def scheduling_constraints(self):
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, SchedulingConstraint)
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
            if node.name == "add":
                type_map[node.name] = node.meta["type"] = node.args[1].meta["type"]
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
        idxc = IndexingContext.current()
        i = 0
        for node in root_graph.nodes:
            if node.name == "tiled_loop":
                self.induction_vars[node.args[0]] = tkl.IndexSymbol("ARG" + str(i))
                self.utils.induction_vars.append(self.induction_vars[node.args[0]])
                i += 1

        # Propagate constraints in root graph and subgraphs.
        for graph in subgraphs.values():
            for node in graph.nodes:
                # If loop nodes, get index from output node
                if "tiled_loop" in node.name:
                    output = list(subgraphs[node.args[2]].nodes)[-1]
                    node.meta["index"] = output.meta["index"]
                    continue

                # For write nodes, see if vector to be written has indices and if so use those
                if "write" in node.name:
                    if "index" in node.args[0].meta:
                        node.meta["index"] = node.args[0].meta["index"]
                        continue

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
                        for constraint in self.wave_constraints:
                            if dim == constraint.dim:
                                node.meta["index"][idx] += constraint.apply()
                if node.name == "mma":
                    c_index = None
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
                            if matrix_type == "C":
                                c_index = arg.meta["index"]
                    # Also add index of result matrix
                    for user in list(node.users.keys()):
                        user.meta["index"] = c_index

    def get_string(self, node: fx.Node, i: int, nested_region: bool) -> str:
        prefix = " "
        nested_region_prefix = "b" if nested_region else ""

        def initialize(prefix: str, nested_region: bool):
            return prefix if not nested_region else prefix + prefix

        if node.op == "placeholder":
            if node.name in self.index_map:
                return self.get_string(node.next, i, nested_region)
            value_prefix = nested_region_prefix if nested_region else ""
            self.index_map[node] = value_prefix + f"{str(i)}"
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
                    self.index_map[node] = value_prefix + f"{str(i)}"
                asm_str += ") {\n"
            return asm_str + self.get_string(node.next, i + 1, nested_region)

        asm_str = initialize(prefix, nested_region)

        typed_node = getNode(node)
        if not isinstance(typed_node, UnknownNode) and not isinstance(
            typed_node, TiledLoop
        ):
            self.index_map[node] = f"{nested_region_prefix}{i}"
            return (prefix if not nested_region else prefix + prefix) + (
                f"%{nested_region_prefix}{i} = {typed_node.custom_string(self.index_map)}"
                + self.get_string(node.next, i + 1, nested_region)
            )

        # TODO: Move the other ops below also to custom nodes
        if "tiled_loop" in node.name:
            if nested_region:
                j = self.parent_id
                self.index_map[node] = f"{j}"
                self.parent = None
                self.parent_id = None
                return self.get_string(node.next, j + 1, False)
            asm_str += f"%{i} = "
            args_str = ""
            for iter_arg, init_arg in self.iter_args_to_init_args.items():
                if init_arg.name not in self.index_map:
                    continue
                self.index_map[iter_arg] = self.index_map[init_arg]
            for j, init_arg in enumerate(node.args[1]):
                args_str += f"%arg{str(j)} = %{self.index_map[init_arg]}, "
                if init_arg in self.init_args_to_iter_args:
                    init_arg = self.init_args_to_iter_args[init_arg]
                self.index_map[init_arg] = f"arg{str(j)}"
            asm_str += f"scf.for (K, iter_args = [{args_str}]) {{\n"
            if "subgraph" in node.kwargs:
                self.subgraphs[typed_node.subgraph_name] = node.kwargs["subgraph"]

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
                    if isinstance(arg, fx.Node):
                        arg_list = [arg]
                    else:
                        arg_list = arg
                    for entry in arg_list:
                        asm_str += f"%{self.index_map[entry]}, "
            if self.parent is not None:
                asm_str += "\n" + initialize(prefix, False) + "}\n"
                return asm_str + self.get_string(self.parent, i + 1, nested_region)
            else:
                asm_str += "\n}\n"
            return asm_str

        asm_str += f"%{i} = {node.name}("
        for entry in node.args:
            asm_str += f"%{self.index_map[entry]},"
        self.index_map[node] = node
        if not "index" in node.meta:
            # TODO: This is a hack for the add
            index = node.args[0].meta["index"]
        else:
            index = node.meta["index"]
        asm_str += f"), indexing: {index}\n"
        return asm_str + self.get_string(node.next, i + 1, nested_region)

    def print(self, trace: CapturedTrace | fx.Graph):
        self.index_map = {}
        self.parent = None
        self.parent_id = None
        if isinstance(trace, CapturedTrace):
            self.subgraphs = trace.region_graph.subgraphs
            root = list(trace.get_root_graph().nodes)[0]
        else:
            assert isinstance(trace, fx.Graph)
            root = list(trace.nodes)[0]
        asm_str = self.get_string(root, 0, False)
        print(asm_str)

    def get_tiled_shape(self, shape: list[IndexExpr]):
        tiled_shape = []
        for dim in shape:
            for constraint in self.workgroup_constraints:
                if dim == constraint.dim:
                    tiled_shape.append(constraint.tile_size)
            for constraint in self.tiling_constraints:
                if dim == constraint.dim:
                    tiled_shape.append(constraint.tile_size)
        return tiled_shape

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
                            tiled_shape = self.get_tiled_shape(shape)
                            type = Memory[*list(tiled_shape) + [SMEM_SPACE, dtype]]
                            alloc_node = AllocSharedNode(
                                root_graph, alloc_shared, tiled_shape, dtype, type
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
                                type=Register[*list(tiled_shape) + [dtype]],
                            )

                            read_shared_node.emit()
                            read_shared_fx = read_shared_node.fx_node
                            read_shared_fx.meta["type"] = read_shared_node.type
                            read_shared_fx.meta["index"] = self.utils.global_to_shared(
                                node.meta["index"]
                            )

                            node.replace_all_uses_with(read_shared_fx)

                            # Create write shared node
                            write_shared_node = WriteSharedNode(
                                graph, write_shared, node, alloc, node.args[1]
                            )
                            write_shared_node.emit()
                            write_shared_fx = write_shared_node.fx_node
                            write_shared_fx.meta["type"] = None
                            # Not sure about the indexing here
                            # See design doc, but I believe we can reuse the original index
                            # with the workgroup component and induction variable removed.
                            write_shared_fx.meta["index"] = read_shared_fx.meta["index"]
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
                return self.global_delay
            else:
                return self.shared_delay
        if "mma" in name:
            return self.mma_delay
        return 0

    def update_stage_index(self, node: fx.Node, stage: int, outside_loop: bool = False):
        # Depending on the stage, we need to remap ARG0
        # For example, if stage = 0, ARG0 -> ARG0
        #              if stage = 1, ARG0 -> ARG0 - 1
        #              if stage = 2, ARG0 -> ARG0 - 2
        #              ...
        # Also, we need to update the index to account for multibuffering.
        # The need for multibuffering comes when we unroll the loop > 1 times.
        # In that case, we can just use the buffer index to determine the
        # batch index for the shared memory alloc.
        if not self.utils.is_shared_memory_read_or_write(
            node
        ) and not self.utils.is_global_memory_read_or_write(node):
            return
        if "index" in node.meta:
            # Update induction variable indices on global reads/writes
            ivar = tkl.IndexSymbol("ARG0")
            if self.utils.is_global_memory_read_or_write(node):
                # We don't need to update the index outside the loop because we
                # do this during codegen
                if not outside_loop:
                    index = [None for _ in range(len(node.meta["index"]))]
                    for i in range(len(node.meta["index"])):
                        index[i] = node.meta["index"][i].replace(ivar, ivar - stage)
                    node.meta["index"] = index
            # Increase dimensionality on shared reads/writes
            if self.utils.is_shared_memory_read_or_write(node) and self.num_buffers > 1:
                batch_index = node.meta["buffer_index"]
                node.meta["index"] = [batch_index] + node.meta["index"]

    def create_loop_graph(self, value_map, iter_args):
        kernel = fx.Graph()
        # Keep track of values by stage
        staged_value_map = {}
        # Keep track of mapping from node to stage
        node_to_stage = {}
        iter_arg_names = [x.name for x in iter_args]

        # Populate staged values with iter args.
        for stage, nodes in self.iter_args.items():
            staged_value_map[stage] = {}
            for node in nodes:
                staged_value_map[stage][node] = node
                node_to_stage[node] = stage

        # Add c_reg iter args
        for stage, nodes in self.nodes_by_stage.items():
            for node in nodes:
                if "mma" in node.name:
                    i, j, k = node.name.split("_")[-3:]
                    if int(k) != self.batch_k - 1:
                        continue
                    for iter_arg in iter_args:
                        if iter_arg.name == f"c_reg_{i}_{j}":
                            node_to_stage[iter_arg] = stage
                            staged_value_map[stage][iter_arg] = iter_arg

        def arg_mapper(stage: int, node: fx.Node):
            # If a node has no stage, it is a placeholder.
            if node not in node_to_stage:
                return value_map[node.name]
            # Check if the node exists in the provided stage.
            if node in staged_value_map[stage]:
                return staged_value_map[stage][node]
            # If not, use the node's current stage.
            stage = node_to_stage[node]
            if node in staged_value_map[stage]:
                return staged_value_map[stage][node]

        def get_stage(target):
            for stage, nodes in self.nodes_by_stage.items():
                for node in nodes:
                    if node == target:
                        return stage

        def find_c_node(node):
            for arg in node.args:
                if "mma" in arg.name:
                    return find_c_node(arg)
                if "c_reg" in arg.name:
                    return arg

        result_map = {}
        for time, nodes in self.nodes_by_time.items():
            for node in nodes:
                if "alloc" in node.name or "c_reg" in node.name:
                    continue
                stage = get_stage(node)
                if stage not in staged_value_map:
                    staged_value_map[stage] = {}
                if "sync" in node.name:
                    # Dependencies already captured by creation of the sync node.
                    continue
                node.meta["stage"] = stage
                new_node = kernel.node_copy(node, partial(arg_mapper, stage))
                self.update_stage_index(new_node, stage)
                # This mapping is constantly updated to only the last reference.
                # Is that always enough?
                node_to_stage[node] = stage
                staged_value_map[stage][node] = new_node
                if node.name in iter_arg_names:
                    result_map[node.name] = new_node
                # Check for mma
                if "mma" in node.name:
                    last_index = int(node.name.split("_")[-1])
                    if last_index == self.batch_k - 1:
                        indices = node.name.split("_")[-3:-1]
                        c_reg_name = "_".join(["c_reg"] + indices)
                        result_map[c_reg_name] = new_node
                        c_reg_node = find_c_node(node)
                        next_stage = 0 if stage == 0 else stage - 1
                        staged_value_map[next_stage][c_reg_node] = new_node
                        node_to_stage[c_reg_node] = next_stage

        mapped_iter_args = []
        for arg in iter_args:
            if arg.name in result_map:
                mapped_iter_args.append(result_map[arg.name])
            else:
                # TODO: This will have to be extended when the number of uses exceeds 2 cases
                mapped_iter_args.append(self.sync_parent[arg])
        kernel.create_node("output", "output", (mapped_iter_args,))
        return kernel

    def update_staged_value_map(
        self,
        value_map: dict[str, fx.Node],
        stage: int,
        name: str,
        node: fx.Node,
        next_stage: int,
    ) -> bool:
        value_map[stage][name] = node
        # Whenever we compute the last mma node in an mma-chain,
        # we need to update the value mapper for the corresponding
        # c_reg.
        if "mma" in name:
            i, j, k = name.split("_")[-3:]
            if int(k) == self.batch_k - 1:
                c_reg_name = "_".join(["c_reg", i, j])
                value_map[next_stage][c_reg_name] = node
                return True
        return False

    def gather_nodes(
        self,
        nodes: list[fx.Node],
        stage: int,
        nodes_by_stage: dict[int, list[fx.Node]],
    ) -> list[fx.Node]:
        """
        Given nodes that correspond to the same time, group them
        by stage, ignoring any c_reg nodes.
        """
        output_nodes = []
        for node in nodes:
            if node in nodes_by_stage[stage]:
                if "c_reg" in node.name:
                    continue
                output_nodes.append(node)
        return output_nodes

    def gather_nodes_by_stage(
        self,
        nodes: list[fx.Node],
        stages: list[int],
        nodes_by_stage: dict[int, list[fx.Node]],
    ) -> list[fx.Node]:
        """
        Given nodes that correspond to the same time, group them
        by stage, ignoring any c_reg nodes.
        """
        nodes_by_stage_per_time = {}
        for node in nodes:
            for stage in stages:
                if node in nodes_by_stage[stage]:
                    if "c_reg" in node.name:
                        continue
                    if stage not in nodes_by_stage_per_time:
                        nodes_by_stage_per_time[stage] = []
                    nodes_by_stage_per_time[stage].append(node)
        return nodes_by_stage_per_time

    def initialize_mma_init_args(
        self, source: fx.Node, expanded_graph: fx.Graph, value_map: dict[str, fx.Node]
    ):
        """
        Initialize c_reg nodes to construct_register_from_metadata
        nodes from the root graph.
        """
        mma_init_args = {}
        for node in expanded_graph.nodes:
            if "c_reg" not in node.name:
                continue
            for src_node in source.args[1]:
                target_name = src_node.name.replace(
                    "construct_register_from_metadata", "c_reg"
                )
                if target_name == node.name:
                    mma_init_args[target_name] = value_map[src_node.name]
        return mma_init_args

    def create_prologue(
        self,
        scheduled_graph: fx.Graph,
        staged_value_map: dict[int, dict[str, fx.Node]],
        mma_init_args: list[fx.Node],
    ):
        def map_stage(stage: int):
            """
            During scheduling, we have stages 0, 1, 2
            and these map to stages 2, 1, 0.
            """
            reversed_stages = list(range(self.max_stage, -1, -1))
            return reversed_stages[stage]

        new_iter_args = []
        old_iter_args = []
        stages = sorted(list(self.nodes_by_stage.keys()))
        time_index_per_stage = {i: 0 for i in range(len(stages))}
        start_time_per_stage = [
            self.initiation_iterval * (self.max_stage - i) for i in range(len(stages))
        ]
        times = list(self.nodes_by_absolute_time.keys())
        max_time_index = len(times)
        for time in self.nodes_by_absolute_time.keys():
            for stage in reversed(self.prologue.keys()):
                if time < start_time_per_stage[stage]:
                    continue
                stage_time = times[time_index_per_stage[stage]]
                nodes = self.nodes_by_absolute_time[stage_time]
                prolog_nodes = self.gather_nodes(nodes, stage, self.prologue)
                # Induction variables at stage k have to be mapped to k - max_stage
                for subnode in prolog_nodes:
                    if "sync" in subnode.name:
                        staged_value_map[stage][subnode.name] = staged_value_map[stage][
                            subnode.args[0].name
                        ]
                        continue
                    new_node = scheduled_graph.node_copy(
                        subnode, lambda node: staged_value_map[stage][node.name]
                    )
                    new_node.name = subnode.name + "_prolog" + str(stage)
                    new_node.meta["stage"] = map_stage(stage)
                    self.update_stage_index(new_node, new_node.meta["stage"], True)
                    next_stage = stage - 1 if stage > 0 else stage
                    updated = self.update_staged_value_map(
                        staged_value_map, stage, subnode.name, new_node, next_stage
                    )
                    if updated:
                        name = "c_reg_" + "_".join(subnode.name.split("_")[-3:-1])
                        mma_init_args[name] = new_node

                time_index_per_stage[stage] = (
                    time_index_per_stage[stage] + 1
                ) % max_time_index

        for stage, nodes in self.iter_args.items():
            for subnode in sorted(list(nodes), key=lambda node: node.name):
                new_iter_args.append(staged_value_map[stage][subnode.name])
                old_iter_args.append(subnode)

        # Add original iter args
        new_iter_args += [mma_init_args[x.name] for x in self.mma_args]
        old_iter_args += self.mma_args
        self.iter_args_to_init_args = {}
        self.init_args_to_iter_args = {}
        for init_arg, iter_arg in zip(new_iter_args, old_iter_args):
            self.iter_args_to_init_args[iter_arg] = init_arg
            self.init_args_to_iter_args[init_arg] = iter_arg

        return new_iter_args, old_iter_args

    def create_scheduled_graph(
        self, expanded_graph: fx.Graph, expanded_root_graph: fx.Graph, scheduler, trace
    ):
        """
        This pass uses the macrokernel graph and the generated schedule
        and constructs a scheduled macrokernel graph that can be used
        for MLIR emission.
        """
        scheduled_graph = fx.Graph()

        value_map = {}
        placeholder_map = {}
        new_tiled_loop = None
        idxc = IndexingContext.current()
        self.num_buffers = (tkl.sym.UNROLL_FACTOR).subs(idxc.subs)

        # Initialize mapping between creg and construct_data_from_register
        def initialize_creg_mapping(staged_value_map, stage: int):
            for node in expanded_graph.nodes:
                if "c_reg" in node.name:
                    i, j = node.name.split("_")[-2:]
                    reg_node_name = "_".join(["construct_register_from_metadata", i, j])
                    staged_value_map[stage][node.name] = value_map[reg_node_name]

        def update_value_map(name: str, node: fx.Node):
            value_map[name] = node
            # Whenever we compute the last mma node in an mma-chain,
            # we need to update the value mapper for the corresponding
            # c_reg.
            if "mma" in name:
                i, j, k = name.split("_")[-3:]
                if int(k) == self.batch_k - 1:
                    c_reg_name = "_".join(["c_reg", i, j])
                    value_map[c_reg_name] = new_node

        def epilogue_arg_mapper(node: fx.Node):
            if node.name in value_map:
                return value_map[node.name]
            return node

        def initialize_staged_value_map(value_map):
            # All nodes in existing value map represent placeholders or shared memory
            # allocations and so are exposed in all stages.
            staged_value_map = {}
            for stage in range(self.max_stage + 1):
                staged_value_map[stage] = {}
                for name, subnode in value_map.items():
                    staged_value_map[stage][name] = subnode
            return staged_value_map

        for node in expanded_root_graph.nodes:
            typed_node = getNode(node)
            if isinstance(typed_node, TiledLoop):
                staged_value_map = initialize_staged_value_map(value_map)
                initialize_creg_mapping(staged_value_map, self.max_stage)
                mma_init_args = self.initialize_mma_init_args(
                    node, expanded_graph, value_map
                )
                init_args, iter_args = self.create_prologue(
                    scheduled_graph, staged_value_map, mma_init_args
                )

                # Emit loop
                new_tiled_loop = scheduled_graph.node_copy(node)
                new_tiled_loop.args = (
                    node.args[0],
                    init_args,
                    node.args[2],
                    [value_map[x.name] for x in node.args[3]],
                )
                # TODO: Figure out how to extend to multiple tiling
                trip_counts = int(
                    self.tiling_constraints[0].trip_counts().subs(idxc.subs)
                )
                loop_body = self.create_loop_graph(placeholder_map, iter_args)
                new_tiled_loop.meta["start"] = (
                    len(self.nodes_by_stage) - 1
                ) * self.num_buffers
                new_tiled_loop.meta["end"] = trip_counts
                new_tiled_loop.meta["step"] = self.num_buffers
                new_tiled_loop.kwargs = {
                    "subgraph": loop_body,
                    "iter_args": iter_args,
                }
                update_value_map(node.name, new_tiled_loop)

                # Emit nodes representing indexing into the list of results of the loop
                for idx, arg in enumerate(iter_args):
                    get_res = GetResultNode(
                        scheduled_graph, get_result, new_tiled_loop, idx
                    )
                    get_res.emit()
                    update_value_map(arg.name, get_res.fx_node)

                def update_sync_nodes(value_map):
                    for child, parent in self.sync_parent.items():
                        value_map[child.name] = value_map[parent.name]

                # Emit epilogue
                for stage in reversed(self.epilogue.keys()):
                    for subnode in self.epilogue[stage]:
                        if "c_reg" in subnode.name:
                            continue
                        new_node = scheduled_graph.node_copy(
                            subnode, epilogue_arg_mapper
                        )
                        new_node.name = subnode.name + "_epilog" + str(stage)
                        new_node.meta["stage"] = (
                            trip_counts - self.num_buffers
                        ) - stage
                        self.update_stage_index(
                            new_node, new_node.meta["stage"] % self.num_buffers, True
                        )
                        update_value_map(subnode.name, new_node)
                    # TODO: Extend to stages > 2.
                    if len(self.epilogue[stage]) > 0:
                        update_sync_nodes(value_map)

                continue

            # Map result nodes from the original graph to the scheduled graph.
            # The result nodes will be the nodes corresponding to c_reg/mma
            if "get_result" in node.name:
                c_reg_name = "_".join(
                    ["c_reg"] + [str(x) for x in node.meta["output_index"]]
                )
                value_map[node.name] = value_map[c_reg_name]
                continue

            new_node = scheduled_graph.node_copy(
                node, lambda node: value_map[node.name]
            )
            if node.op == "placeholder" or "alloc" in node.name:
                placeholder_map[node.name] = new_node
            update_value_map(node.name, new_node)

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
        # TODO: This is hardcoded and should be deduced based on dimension mappings.
        waves_m, waves_n, _ = hardware_constraint.waves_per_block
        for sym, val in idxc.frozen_subs:
            if sym.name == "BLOCK_M":
                block_m = val
            if sym.name == "BLOCK_N":
                block_n = val
            if sym.name == "BLOCK_K":
                block_k = val
        self.batch_m = (block_m // waves_m) // mma_m
        self.batch_n = (block_n // waves_n) // mma_n
        self.batch_k = block_k // mma_k
        repeat_times = {
            "M": self.batch_m,
            "N": self.batch_n,
            "K": self.batch_k,
            "BLOCK_M": self.batch_m,
            "BLOCK_N": self.batch_n,
            "BLOCK_K": self.batch_k,
        }

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
                    if "index" in node.meta:
                        mma_tile_sizes = self.utils.get_mma_tile_sizes(new_node)
                        new_node.meta["index"] = [
                            new_node.meta["index"][0] + sympy.Mul(i, mma_tile_sizes[0]),
                            new_node.meta["index"][1] + sympy.Mul(j, mma_tile_sizes[1]),
                        ]

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
                    # Inputs are not duplicated
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
                    for arg in node.all_input_nodes:
                        if arg.meta["type"] is not None:
                            node_type = arg.meta["type"]
                            break
                if len(node_type.symbolic_shape) == 2:
                    repeat0, repeat1 = [
                        repeat_times[x.name] for x in node_type.symbolic_shape
                    ]
                elif len(node_type.symbolic_shape) == 1:
                    repeat0 = repeat_times[node_type.symbolic_shape[0].name]
                    repeat1 = 0
                else:
                    raise ValueError("Only 1D and 2D shapes supported.")

                duplicate_node(repeat0, repeat1, node)

        expanded_root_graph = fx.Graph()
        duplicate_map = {}

        def duplicate_root_node(
            m: int, k: int, node: fx.Node, loop_results: list[fx.Node], duplicates_map
        ):
            def arg_mapper(suffix: str):
                def _(node: fx.Node):
                    if not hasattr(arg_mapper, "i"):
                        arg_mapper.i = 0
                    if "tiled_loop" in node.name:
                        result = loop_results[arg_mapper.i]
                        arg_mapper.i = (arg_mapper.i + 1) % len(loop_results)
                        return result
                    if node.op == "placeholder" or "alloc" in node.name:
                        return node

                    if node in duplicates_map:
                        for duplicate in duplicate_map[node]:
                            if duplicate.name.endswith(suffix):
                                return duplicate

                    raise Exception("Could not find a valid mapping during expansion.")

                return _

            duplicates = []
            m = max(m, 1)
            k = max(k, 1)
            for i in range(m):
                for j in range(k):
                    # The arg_mapper here does not correctly map the write_shared
                    # before the add the the corresponding read.
                    suffix = index_suffix(i, j)
                    new_node = expanded_root_graph.node_copy(node, arg_mapper(suffix))
                    new_node.name = node.name + suffix
                    duplicates.append(new_node)

                    mma_tile_sizes = self.utils.get_mma_tile_sizes(new_node)
                    if "index" in node.meta:
                        old_index = new_node.meta["index"]
                        # For now special case for values which are indexed only in one dimension
                        # TODO: Needs to be modified to handle 32x32x8 instruction
                        if len(old_index) == 1:
                            if m == 1:
                                new_node.meta["index"] = [
                                    old_index[0] + sympy.Mul(i, 16)
                                ]
                            elif k == 1:
                                new_node.meta["index"] = [
                                    old_index[0] + sympy.Mul(j, 16)
                                ]
                            elif "add" in node.name:
                                # TODO: Hack for add
                                new_node.meta["index"] = [
                                    old_index[0] + sympy.Mul(j, 16)
                                ]
                            else:
                                raise Exception("Invalid indexing")
                        else:
                            new_node.meta["index"] = [
                                old_index[0] + sympy.Mul(i, mma_tile_sizes[0]),
                                old_index[1] + sympy.Mul(j, mma_tile_sizes[1]),
                            ]
            return duplicates

        loop_results = []
        for node in root_graph.nodes:
            node_type = node.meta["type"]
            if "tiled_loop" in node.name:
                new_node = expanded_root_graph.node_copy(node)
                # Update iter_args if they have been duplicated
                new_iter_args = []
                for iter_arg in new_node.args[1]:
                    if iter_arg in duplicate_map:
                        new_iter_args += duplicate_map[iter_arg]
                if len(new_iter_args) > 0:
                    new_node.args = tuple(
                        x if i != 1 else new_iter_args for i, x in enumerate(node.args)
                    )
                    for idx, arg in enumerate(new_iter_args):
                        get_res = GetResultNode(
                            expanded_root_graph, get_result, new_node, idx
                        )
                        get_res.emit()
                        get_res.fx_node.meta["output_index"] = [
                            idx // self.batch_n,
                            idx % self.batch_n,
                        ]
                        loop_results.append(get_res.fx_node)
                continue

            if (
                node.op == "placeholder"
                or "alloc" in node.name
                or "output" in node.name
            ):
                expanded_root_graph.node_copy(node)
                continue
            if node_type is None:
                # If type not available, must be a write, so get it
                # from args.
                for arg in node.all_input_nodes:
                    if arg.meta["type"] is not None:
                        node_type = arg.meta["type"]
                        break
            if len(node_type.symbolic_shape) == 2:
                repeat_0, repeat_1 = [
                    repeat_times[x.name] for x in node_type.symbolic_shape
                ]
            elif len(node_type.symbolic_shape) == 1:
                repeat_0 = repeat_times[node_type.symbolic_shape[0].name]
                repeat_1 = 0
                # TODO: This does not quite work yet
                # if "add" in node.name and (
                #     (
                #         (type := node.args[0].meta["type"]) is not None
                #         and len(type.symbolic_shape) > 1
                #     )
                #     or (
                #         (type := node.args[1].meta["type"]) is not None
                #         and len(type.symbolic_shape) > 1
                #     )
                # ):
                #     shape = type.symbolic_shape
                #     if len(shape) == 2:
                #         repeat_1 = repeat_times[shape[1].name]
            else:
                raise ValueError("Only 1D and 2D shapes supported.")
            duplicates = duplicate_root_node(
                repeat_0, repeat_1, node, loop_results, duplicate_map
            )
            duplicate_map[node] = duplicates

        return expanded_graph, expanded_root_graph

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
            i, j, _ = node.name.split("_")[-3:]
            for node in graph.nodes:
                if self.utils.is_creg_with_indices(i, j, node):
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
                if "128" not in node.name:
                    i, j = node.name.split("_")[-2:]
                    prefix = "_".join(node.name.split("_")[:-2]).replace(
                        "write", "read"
                    )
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
                else:
                    i, k0, k1 = node.name.split("_")[-3:]
                    name = node.name.replace("localwrite128_", "")
                    prefix = "_".join(name.split("_")[:-3]).replace("write", "read")
                    kvalues = [k0, k1]
                    for u in range(2):
                        read_shared_name = prefix + "_" + str(i) + "_" + str(kvalues[u])
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
        for constraint in self.scheduling_constraints:
            for unit, resource in constraint.resources.items():
                match unit:
                    case "GLOBAL":
                        resourceVector[0] = resource
                    case "SHARED":
                        resourceVector[1] = resource
                    case "MMA":
                        resourceVector[2] = resource

        scheduler = ms.ModuloScheduler(resourceVector, self.dependenceGraph)
        scheduler.generateSchedule()
        return scheduler

    def print_schedule(self, file_name):

        def print_formatted_line(i, row, line, prefix=""):
            fmt_str = "{:<5}"
            if i == 0:
                row = [prefix + str(time)] + row
            else:
                row = [""] + row
            for _ in range(len(stages)):
                fmt_str += "|{:^38}"
            line += fmt_str.format(*row)
            line += "\n"
            return line

        def add_line_divider(line):
            for _ in range(120):
                line += "="
            line += "\n"
            return line

        def print_subschedule(nodes_by_stage_per_time, line, prefix=""):
            i = 0
            done = False
            while not done:
                node_row = []
                for stage in stages:
                    if stage not in nodes_by_stage_per_time:
                        nodes_by_stage_per_time[stage] = []
                    if len(nodes_by_stage_per_time[stage]) == 0:
                        node_row.append("")
                        continue
                    node = nodes_by_stage_per_time[stage][0]
                    nodes_by_stage_per_time[stage].pop(0)
                    node_row.append(node.name)

                line = print_formatted_line(i, node_row, line, prefix)
                i += 1

                done = True
                for stage in stages:
                    if len(nodes_by_stage_per_time[stage]) > 0:
                        done = False
                        break

            line += "\n"
            return line

        stages = sorted(list(self.nodes_by_stage.keys()))
        line = ""
        for time, nodes in self.nodes_by_time.items():
            line = add_line_divider(line)
            nodes_by_stage_per_time = self.gather_nodes_by_stage(
                nodes, stages, self.nodes_by_stage
            )
            line = print_subschedule(nodes_by_stage_per_time, line)

        prolog_line = ""
        epilog_line = ""
        start_times = {
            i: self.initiation_iterval * (self.max_stage - i)
            for i in range(len(stages))
        }
        time_indices_per_stage = {i: 0 for i in range(len(stages))}
        times = list(self.nodes_by_absolute_time.keys())
        for time, nodes in self.nodes_by_absolute_time.items():
            prolog_line = add_line_divider(prolog_line)
            epilog_line = add_line_divider(epilog_line)

            prolog_nodes_by_stage_per_time = {}
            for stage, start_time in start_times.items():
                if time < start_time:
                    prolog_nodes_by_stage_per_time[stage] = []
                    continue
                stage_time = times[time_indices_per_stage[stage]]
                prolog_nodes_by_stage_per_time[stage] = self.gather_nodes(
                    self.nodes_by_absolute_time[stage_time], stage, self.prologue
                )
            prolog_line = print_subschedule(
                prolog_nodes_by_stage_per_time, prolog_line, "P"
            )
            for stage, start_time in start_times.items():
                if time >= start_time:
                    time_indices_per_stage[stage] += 1

            epilog_nodes_by_stage_per_time = self.gather_nodes_by_stage(
                nodes, stages, self.epilogue
            )
            epilog_line = print_subschedule(
                epilog_nodes_by_stage_per_time, epilog_line, "E"
            )

        with open(file_name, "w") as f:
            f.write(prolog_line)
            f.write(line)
            f.write(epilog_line)

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
        self.initiation_iterval = len(scheduler.RT)
        sorted_schedule = dict(sorted(scheduler.schedule.items(), key=lambda x: x[1]))

        # Partition graph by stage and go back to using the fx.Nodes.
        self.nodes_by_stage = {}
        self.nodes_by_time = {}
        self.nodes_by_absolute_time = {}
        self.mma_args = []
        for node, t in sorted_schedule.items():
            inverse_node = self.inverse_mapper[node]
            if "c_reg" in inverse_node.name:
                self.mma_args.append(inverse_node)
                continue
            stage = t // self.initiation_iterval
            time = t % self.initiation_iterval
            if stage not in self.nodes_by_stage:
                self.nodes_by_stage[stage] = []
            if time not in self.nodes_by_time:
                self.nodes_by_time[time] = []
            if t not in self.nodes_by_absolute_time:
                self.nodes_by_absolute_time[t] = []
            if criteria(inverse_node):
                self.nodes_by_stage[stage].append(inverse_node)
                self.nodes_by_time[time].append(inverse_node)
                self.nodes_by_absolute_time[t].append(inverse_node)
            self.max_stage = stage
        self.nodes_by_time = dict(sorted(self.nodes_by_time.items()))
        self.nodes_by_absolute_time = dict(sorted(self.nodes_by_absolute_time.items()))
        self.mma_args = self.mma_args[::-1]

        self.iter_args = {stage: set() for stage in range(self.max_stage + 1)}
        for stage in self.nodes_by_stage.keys():
            for node in self.nodes_by_stage[stage]:
                for arg in node.all_input_nodes:
                    if (
                        not arg in self.nodes_by_stage[stage]
                        and criteria(arg)
                        and not "c_reg" in arg.name
                    ):
                        self.iter_args[stage].add(arg)

        # Handle scenarios where the use of an iter arg crosses more than
        # one stage boundary.

        # Determines whether the use of a node should be replaced.
        # We only replaces nodes whose stage is greater than the
        # base stage.
        def callback(base_stage: int, replacement: fx.Node, target: fx.Node):
            if target == replacement:
                return False
            for stage, nodes in self.nodes_by_stage.items():
                for node in nodes:
                    if target == node:
                        return stage > base_stage
            return False

        def find_node_stage(target: fx.Node):
            for stage, nodes in self.nodes_by_stage.items():
                for node in nodes:
                    if target == node:
                        return stage

        def find_node_time(target: fx.Node):
            for time, nodes in self.nodes_by_time.items():
                for node in nodes:
                    if target == node:
                        return time

        self.sync_parent = {}
        for stage_i in range(0, self.max_stage):
            for arg_i in self.iter_args[stage_i]:
                for stage_j in range(stage_i + 1, self.max_stage + 1):
                    if arg_i in self.iter_args[stage_j]:
                        # Create new node
                        graph = arg_i.graph
                        graph.inserting_after(arg_i)
                        sync_node = SyncNode(graph, sync, arg_i)
                        sync_node.emit()
                        new_arg: fx.Node = sync_node.fx_node
                        new_arg.name = arg_i.name + f"_sync_{stage_j}"
                        arg_i.replace_all_uses_with(
                            new_arg,
                            partial(callback, stage_i, new_arg),
                            propagate_meta=True,
                        )
                        self.iter_args[stage_j].remove(arg_i)
                        self.iter_args[stage_j].add(new_arg)
                        arg_i_stage = find_node_stage(arg_i)
                        self.nodes_by_stage[arg_i_stage].append(new_arg)
                        arg_i_time = find_node_time(arg_i)
                        self.nodes_by_time[arg_i_time].append(new_arg)
                        self.sync_parent[new_arg] = arg_i

        # Create prologue and epilogue after sync nodes have been created.
        self.prologue = {stage: [] for stage in range(self.max_stage + 1)}
        self.epilogue = {stage: [] for stage in range(self.max_stage + 1)}
        for stage in self.prologue.keys():
            for i in range(0, stage):
                self.prologue[stage] += self.nodes_by_stage[i]
        for stage in self.epilogue.keys():
            for i in range(stage + 1, self.max_stage + 1):
                self.epilogue[stage] += self.nodes_by_stage[i]

        self.print_schedule("schedule.csv")

    def insert_barriers(self, graph: fx.Graph):
        """
        This function inserts barrier nodes into the graph following a very
        simple approach - if a write to shared memory is followed by a read
        from shared memory and vice versa, we insert a barrier node in between.
        """

        has_read_shared_ancestor = False
        has_write_shared_ancestor = False
        for node in graph.nodes:
            typed_node = getNode(node)
            if node.next is None:
                continue
            if isinstance(typed_node, WriteSharedNode):
                has_write_shared_ancestor = True
                if has_read_shared_ancestor:
                    graph.inserting_before(node)
                    barrier_node = BarrierNode(graph, barrier)
                    barrier_node.emit()
                    has_read_shared_ancestor = False
            if isinstance(typed_node, ReadSharedNode):
                has_read_shared_ancestor = True
                if has_write_shared_ancestor:
                    graph.inserting_before(node)
                    barrier_node = BarrierNode(graph, barrier)
                    barrier_node.emit()
                    has_write_shared_ancestor = False
            ## read after write barrier
            # if isinstance(typed_node, ReadSharedNode) and isinstance(
            #    getNode(node.next), WriteSharedNode
            # ):
            #    graph.inserting_after(node)
            #    barrier_node = BarrierNode(graph, barrier)
            #    barrier_node.emit()
            ## write after read barrier
            # elif isinstance(typed_node, WriteSharedNode) and isinstance(
            #    getNode(node.next), ReadSharedNode
            # ):
            #    graph.inserting_after(node)
            #    barrier_node = BarrierNode(graph, barrier)
            #    barrier_node.emit()
            elif isinstance(typed_node, TiledLoop):
                # recurse into loop body
                self.insert_barriers(node.kwargs["subgraph"])

    def handle_larger_global_loads(self, graph):
        """
        Given a graph where the global load, shared store and load are of the
        same size, this function modifies the graph so that the global load
        can be of 128bits while the shared store and load can be either
        64 or 128 bits.
        """
        idxc = IndexingContext.current()
        global_load_elems = tkl.sym.GLOBAL_LOAD_ELEMS_PER_THREAD.subs(idxc.subs)
        if global_load_elems != 8:
            return graph

        def find_node(matrix, target_i, target_k):
            for user in matrix.users.keys():
                try:
                    i, k = [int(x) for x in user.name.split("_")[-2:]]
                except:
                    continue
                if i == target_i and k == target_k:
                    return user

        def find_node_by_name(name):
            for node in graph.nodes:
                if node.name == name:
                    return node

        # We have read nodes of type "read_i_k" from A and "read_j_k" from B.
        # We will combine read_i_k/write_shared_i_k and read_i_k+1/write_shared_i_k+!
        # into a read_i_k_k+1/write_shared_i_k_k+1. We also update the indices of
        # the reads and write_shareds to account for the offset.
        # So thread reading from col 0 -> 0
        #    thread reading from col 4 -> 8
        #    thread reading from col 8 -> 16
        # We do this by doubling the thread offset.
        processed = []

        def matching_index(k):
            if k % 2 == 0:
                return k + 1
            return k - 1

        for node in graph.nodes:
            if "read" in node.name and not "shared" in node.name:
                if "globalload128" in node.name:
                    continue
                i, k = [int(x) for x in node.name.split("_")[-2:]]
                matrix = node.all_input_nodes[0]
                if (matrix, i, k) in processed:
                    continue
                matching_node = find_node(matrix, i, matching_index(k))
                processed.append((matrix, i, k))
                processed.append((matrix, i, matching_index(k)))
                meta = None
                if matching_index(k) > k:
                    graph.inserting_after(matching_node)
                    meta = node.meta
                else:
                    graph.inserting_after(node)
                    meta = matching_node.meta
                # Create read node
                new_name = "_".join(
                    node.name.split("_")[:-2]
                    + ["globalload128"]
                    + [str(x) for x in [i] + sorted([k, matching_index(k)])]
                )
                read_node = graph.create_node(
                    "call_function",
                    target=read,
                    args=(node.args[0], tkl.sym.GLOBAL_LOAD_ELEMS_PER_THREAD, None),
                    name=new_name,
                    kwargs={},
                )
                # RAUW
                read_node.meta = meta
                # Modify read index to account for the increased offset
                thread_offset = read_node.meta["index"][1].subs(
                    {tkl.IndexSymbol("ARG0"): 0, tkl.sym.MMA_K: 0}
                )
                read_node.meta["index"][1] += thread_offset
                node.replace_all_uses_with(read_node)
                matching_node.replace_all_uses_with(read_node)
                graph.erase_node(node)
                graph.erase_node(matching_node)

                # Combine both users into a single write_shared_node
                users = []
                alloc = None
                for user in read_node.users.keys():
                    if "write_shared" in user.name:
                        users.append(user)
                        alloc = user.args[1]
                graph.inserting_after(read_node)
                new_name = "_".join(
                    users[0].name.split("_")[:-2]
                    + ["localwrite128"]
                    + [str(x) for x in [i] + sorted([k, matching_index(k)])]
                )
                write_shared_node = graph.create_node(
                    "call_function",
                    target=write_shared,
                    args=(read_node, alloc, tkl.sym.GLOBAL_LOAD_ELEMS_PER_THREAD),
                    name=new_name,
                    kwargs={},
                )
                write_shared_node.meta = deepcopy(users[0].meta)
                write_shared_node.meta["index"] = self.utils.global_to_shared(
                    read_node.meta["index"]
                )
                for user in users:
                    user.replace_all_uses_with(write_shared_node)
                    graph.erase_node(user)

        return graph

    def unroll_graph(
        self, graph: fx.Graph, root_graph: fx.Graph, unroll_factor_expr: IndexExpr
    ) -> fx.Graph:
        """
        This function unrolls the given graph N times, handling any dependencies
        between the loop iterations and ensuring that writes/reads to shared memory
        happen to separate buffers.
        """
        idxc = IndexingContext.current()
        unroll_factor = unroll_factor_expr.subs(idxc.subs)
        if unroll_factor == 1:
            return graph
        unrolled_graph = fx.Graph()
        val_map = {}
        for node in graph.nodes:
            for arg in node.all_input_nodes:
                if arg.op == "placeholder" and arg not in graph.nodes:
                    val_map[arg] = arg
                if self.utils.is_shared_memory_alloc(arg):
                    val_map[arg] = arg
                    root_arg = self.utils.get_node_from_root(arg, root_graph)
                    root_arg.meta["num_buffers"] = unroll_factor
        for i in range(unroll_factor):
            graph_map = dict(val_map)
            output = unrolled_graph.graph_copy(graph, graph_map)
            for original, new in graph_map.items():
                if original not in graph.nodes:
                    continue
                new.name = f"mve{i}_" + original.name
                new.meta["buffer_index"] = i
                if i == 0:
                    continue
                ivar = tkl.IndexSymbol("ARG0")
                if "index" in new.meta:
                    if self.utils.is_global_memory_read_or_write(new):
                        new_index = [None for _ in range(len(new.meta["index"]))]
                        for i in range(len(new.meta["index"])):
                            new_index[i] = new.meta["index"][i].replace(ivar, ivar + 1)
                        new.meta["index"] = new_index
        # Connect the loop carried dependencies
        for i in range(1, unroll_factor):
            for node in unrolled_graph.nodes:
                if self.utils.is_c_reg_at_stage(node, i):
                    mma_node = self.utils.get_mma_node_at_stage_with_k_index(
                        i - 1, self.batch_k - 1, node, unrolled_graph
                    )
                    node.replace_all_uses_with(mma_node)
                    unrolled_graph.erase_node(node)
        # Rename creg nodes
        for node in unrolled_graph.nodes:
            if "c_reg" in node.name:
                node.name = node.name.replace("mve0_", "")
        unrolled_graph.create_node("output", "output", (output,))
        return unrolled_graph

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

        # Do shared memory promotion if required
        self.promote_to_shared_memory(trace, idxc)

        # Create "macrokernel" graph
        expanded_graph, expanded_root_graph = self.create_expanded_graph(trace, idxc)
        # Do graph rewrites in case of different load/store sizes from/to global/lds
        expanded_graph = self.handle_larger_global_loads(expanded_graph)
        # Unroll loop N times for multi-buffering
        unrolled_graph = self.unroll_graph(
            expanded_graph, expanded_root_graph, tkl.sym.UNROLL_FACTOR
        )
        # Schedule "macrokernel" graph
        scheduler = self.construct_schedule(unrolled_graph)
        # Construct prologue and epilogue
        self.construct_prologue_and_epilogue(scheduler)
        scheduled_graph = self.create_scheduled_graph(
            expanded_graph, expanded_root_graph, scheduler, trace
        )
        # Insert Barriers
        self.insert_barriers(scheduled_graph)

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
        workgroup_size = self.hardware_constraints[0].get_threads_per_block()
        for i in range(len(workgroup_size)):
            if isinstance(workgroup_size[i], IndexExpr):
                workgroup_size[i] = workgroup_size[i].subs(idxc.subs)
        subgroup_size = self.hardware_constraints[0].threads_per_wave
        dispatch_entrypoint = exe.define_entrypoint(
            entrypoint_name, kernel_sig, grid, workgroup_size, subgroup_size
        )
        emitter = WaveEmitter(dispatch_entrypoint, trace)
        emitter.mma_matrix_shapes = self.hardware_constraints[0].mma_matrix_shapes()
        emitter.acc_vector_shape = self.hardware_constraints[0].get_vector_shape("C")
        emitter.offset_fn = self.hardware_constraints[0].offset_gpr_c

        self.print(scheduled_graph)

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

        output_name = "mma.mlir"
        reference_name = "reference.mlir"
        with open(output_name, "w") as f:
            f.write(mb.module_op.get_asm())

        try:
            with open(reference_name, "r") as reference_f:
                with open(output_name, "r") as mma_f:
                    ref = reference_f.readlines()
                    mma = mma_f.readlines()
                    diff = list(
                        difflib.unified_diff(
                            mma, ref, fromfile="new", tofile="reference", lineterm=""
                        )
                    )
                    if len(diff) == 0:
                        print("identical to reference")
                    else:
                        print("differences to reference:")
                        for line in diff:
                            print(line)
        except FileNotFoundError:
            print(f"No reference output found, consider creating {reference_name}")

    def aot_execute(self, args, kwargs):
        assert isinstance(launch_context, AOTLaunchContext)

        module = launch_context.module

        mb, exe, kernel_sig, entrypoint_name = self._trace_and_get_kernel_signature(
            args, kwargs, context=module.context, module_op=module.operation
        )

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"

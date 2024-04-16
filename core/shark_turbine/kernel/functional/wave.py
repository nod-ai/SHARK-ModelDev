from typing import Type, Callable, Optional, Dict

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

            @builtin_d.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
            def transform_module():
                @named_sequence("__transform_main", [any_op_t()], [])
                def sequence(target: any_op_t()):

                    @apply_patterns(target)
                    def patterns():
                        transform_d.apply_patterns_canonicalization()

                    loops = structured_transform_ops.structured_match(
                        any_op_t(), target, ops=["scf.for"]
                    )
                    transform_d.apply_licm(loops)
                    # transform_d.apply_cse(target)

            transform_interpreter.apply_named_sequence(
                module,
                transform_module.regions[0].blocks[0].operations[0],
                transform_module,
            )

    def lower_module(self, module: Operation):
        with module.context, Location.unknown():

            @builtin_d.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
            def transform_module():
                @named_sequence("__transform_main", [any_op_t()], [])
                def sequence(target: any_op_t()):
                    target = transform_d.ApplyRegisteredPassOp(
                        any_op_t(), target, "convert-scf-to-cf"
                    )
                    # TODO: This one does not work as one of our functions does
                    #       not respect IREE conventions:
                    #       public functions on executables must be () -> ()

                    # transform_d.ApplyRegisteredPassOp(
                    #     any_op_t(), target, "iree-convert-to-llvm"
                    # )

            transform_interpreter.apply_named_sequence(
                module,
                transform_module.regions[0].blocks[0].operations[0],
                transform_module,
            )

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

    def propagate_types_in_graph(
        self, graph: fx.Graph, type_map: Dict[str, Type], subgraphs: Dict[str, fx.Node]
    ):
        def look_for_type(node: fx.Node) -> Type:
            for input in node.all_input_nodes:
                if input.name in type_map:
                    return type_map[input.name]
            return None

        for node in graph.nodes:
            if node.op == "placeholder":
                if node.name in type_map:
                    node.meta["type"] = type_map[node.name]
                    continue
                node.meta["type"] = type_map[node.name] = node.type
            if node.name == "construct_register_from_metadata":
                args = [x for x in node.args[0]] + [node.args[1]]
                type_map[node.name] = node.meta["type"] = Register[*args]
            if "write" in node.name or "read" in node.name:
                arg_type = look_for_type(node)
                if arg_type is not None:
                    type_map[node.name] = node.meta["type"] = arg_type
            if "subgraph" in node.kwargs:
                subgraph = subgraphs[node.kwargs["subgraph"]]
                implicit_capture_nodes = []
                if "implicit_capture" in node.kwargs:
                    implicit_capture_nodes += node.kwargs["implicit_capture"]
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

    def get_string(self, node: fx.Node, i: int, nested_region: bool):
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
        if "construct_register_from_metadata" in node.name:
            shape, dtype, value = node.args
            simt_shape = None
            if "simt_shape" in node.meta:
                simt_shape = node.meta["simt_shape"]
            asm_str += f"%{i} = construct_register_from_metadata [value = {value}] -> Register<{shape}, {dtype}> -> Register<{simt_shape}, {dtype}>\n"
            self.index_map[node.name] = f"{i}"
            return asm_str + self.get_string(node.next, i + 1, nested_region)
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
            first_node = list(self.subgraphs[node.kwargs["subgraph"]].nodes)[0]
            self.parent = node
            self.parent_id = i
            return asm_str + self.get_string(first_node, 0, True)
        if "alloc" in node.name:
            shape = node.args[0]
            dtype = node.args[1]
            asm_str += f"%{nested_region_prefix}{i} = alloc : Memory<{shape}, {dtype}, #shared>\n"
            self.index_map[node.name] = f"{nested_region_prefix}{i}"
            return asm_str + self.get_string(node.next, i + 1, nested_region)
        if "read" in node.name:
            type = node.args[0].meta["type"]
            shape = type.symbolic_shape
            dtype = type.dtype
            simt_shape = node.args[1]
            asm_str += f"%{nested_region_prefix}{i} = {node.name} %{self.index_map[node.args[0].name]} -> Register<{shape}, {dtype}> -> Register<{simt_shape}, {dtype}>\n"
            self.index_map[node.name] = f"{nested_region_prefix}{i}"
            return asm_str + self.get_string(node.next, i + 1, nested_region)
        if "mma" in node.name:
            lhs = node.args[0]
            lhs_type = lhs.meta["type"]
            lhs_shape = self.hardware_constraints[0].get_vector_shape("A")
            rhs = node.args[1]
            rhs_type = rhs.meta["type"]
            rhs_shape = self.hardware_constraints[0].get_vector_shape("B")
            acc = node.args[2]
            acc_type = acc.meta["type"]
            acc_shape = self.hardware_constraints[0].get_vector_shape("C")
            asm_str += f"%{nested_region_prefix}{i} = mma %{self.index_map[lhs.name]}, %{self.index_map[rhs.name]}, %{self.index_map[acc.name]} : "
            asm_str += f"Register<{lhs_shape}, {lhs_type.dtype}>, Register<{rhs_shape}, {rhs_type.dtype}> -> Register<{acc_shape}, {acc_type.dtype}>\n"
            self.index_map[node.name] = f"{nested_region_prefix}{i}"
            return asm_str + self.get_string(node.next, i + 1, nested_region)
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
        if "write" in node.name:
            memory_type = node.args[1].meta["type"]
            memory_shape = memory_type.symbolic_shape
            memory_dtype = memory_type.dtype
            asm_str += (
                f"%{nested_region_prefix}{i} = {node.name} %{self.index_map[node.args[0].name]}, %{self.index_map[node.args[1].name]}"
                + f" : Memory<{memory_shape}, {memory_dtype}>\n"
            )
            return asm_str + self.get_string(node.next, i + 1, nested_region)
        return asm_str

    def print(self, trace: CapturedTrace):
        self.index_map = {}
        self.subgraphs = trace.region_graph.subgraphs
        self.parent = None
        self.parent_id = None
        root = list(trace.get_root_graph().nodes)[0]
        asm_str = self.get_string(root, 0, False)
        print(asm_str)

    @staticmethod
    def handle_alloc_shared(emitter: WaveEmitter, node: fx.Node):
        try:
            shape, dtype = node.args
        except ValueError as e:
            raise ValidationError("Malformed arguments") from e
        memref_shape = cast_py_literal(emitter, shape)
        element_type = IrType.parse(dtype.ir_type_asm())
        address_space = IntegerAttr.get(IndexType.get(), gpu_d.AddressSpace.Workgroup)
        memref_type = MemRefType.get(memref_shape, element_type, None, address_space)
        alloc = memref_d.alloc(memref_type, [], [])
        emitter.bind_node_proxy(node, IRProxyValue(alloc))

    @staticmethod
    def handle_read_shared(emitter: WaveEmitter, node: fx.Node):
        handle_read(emitter, node)

    @staticmethod
    def handle_write_shared(emitter: WaveEmitter, node: fx.Node):
        handle_write(emitter, node)

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
        new_ops = []
        for graph in subgraphs.values():
            for node in graph.nodes:
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
                            alloc = root_graph.create_node(
                                "call_function",
                                self.handle_alloc_shared,
                                args=(
                                    shape,
                                    dtype,
                                ),
                                kwargs=None,
                                name="alloc_shared",
                            )
                            new_ops.append(self.handle_alloc_shared)
                            alloc.meta["type"] = Memory[
                                *list(shape) + [SMEM_SPACE, dtype]
                            ]
                            current = root_graph._root
                            while current.next.op == "placeholder":
                                current = current.next
                            current.append(alloc)
                            # Create read shared node
                            read_shared = graph.create_node(
                                "call_function",
                                self.handle_read_shared,
                                args=(
                                    alloc,
                                    node.args[1],
                                ),
                                kwargs=None,
                                name="read_shared",
                            )
                            new_ops.append(self.handle_read_shared)
                            read_shared.meta["type"] = Register[*list(shape) + [dtype]]
                            node.replace_all_uses_with(read_shared)
                            # Create write shared node
                            write_shared = graph.create_node(
                                "call_function",
                                self.handle_write_shared,
                                args=(
                                    node,
                                    alloc,
                                    node.args[1],
                                ),
                                kwargs=None,
                                name="write_shared",
                            )
                            new_ops.append(self.handle_write_shared)
                            write_shared.meta["type"] = None
                            node.append(write_shared)
                            write_shared.append(read_shared)

        return new_ops

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
        new_ops = self.promote_to_shared_memory(trace, idxc)

        # Propagate constraints to all nodes in the graph
        self.propagate_constraints(trace)

        # MLIR-style debug print of the graph
        self.print(trace)

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

        # Add handlers for new ops that we introduced during promotion
        for target_func in new_ops:
            WaveEmitter.OP_HANDLERS[target_func] = target_func

        emitter.emit()
        emitter.finish()

        self.canonicalize_module(mb.module_op)
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

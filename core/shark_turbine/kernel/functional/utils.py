# Common Utilities
from typing import Type, Callable, Optional, Dict
import inspect
import math
from functools import partial
import sympy
import difflib

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
    "Utils",
]


class Utils:
    def __init__(self, workgroup_ids, induction_vars) -> None:
        self.workgroup_ids = workgroup_ids
        self.induction_vars = induction_vars

    # Convert Global Memory Index to Shared Memory Index
    def global_to_shared(self, indices: list[IndexExpr]) -> list[IndexExpr]:
        substitutions = {x: 0 for x in self.workgroup_ids}
        substitutions.update({x: 0 for x in self.induction_vars})
        indices = [x.subs(substitutions) for x in indices]
        return indices

    # Gets the tile sizes corresponding to the dimensions of the node
    def get_mma_tile_sizes(self, node: fx.Node) -> list[IndexExpr]:
        mma_tile_sizes = []
        node_type = node.meta["type"]
        if node_type is None:
            for arg in node.all_input_nodes:
                if "type" in arg.meta:
                    node_type = arg.meta["type"]
                    break
        for dim in node_type.symbolic_shape:
            if dim == tkl.sym.M or dim == tkl.sym.BLOCK_M:
                mma_tile_sizes.append(tkl.sym.MMA_M)
            if dim == tkl.sym.N or dim == tkl.sym.BLOCK_N:
                mma_tile_sizes.append(tkl.sym.MMA_N)
            if dim == tkl.sym.K or dim == tkl.sym.BLOCK_K:
                mma_tile_sizes.append(tkl.sym.MMA_K)
        return mma_tile_sizes

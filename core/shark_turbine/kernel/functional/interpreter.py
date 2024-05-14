# Interpreter to check the functional correctness of IR being generated using tkf

from typing import Type, Callable, Optional, Dict
from sys import argv

import inspect
import re
import math
from functools import partial
import sympy

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
    amdgpu_d,
    builtin_d,
    Context,
    InsertionPoint,
    IrType,
    IndexType,
    VectorType,
    Location,
    Module,
    Operation,
    flow_d,
    func_d,
    gpu_d,
    scf_d,
    transform_d,
    vector_d,
    memref_d,
    UnitAttr,
    MemRefType,
    IntegerAttr,
    IndexType,
    arith_d,
    stream_d,
    F32Type,
    F16Type,
)


class Interpreter:
    def __init__(self) -> None:
        pass

    def interpret(self, file_name: str):
        with Context() as ctx:
            with open(file_name, "r") as f:
                asm_str = f.read()
            self.m = Module.parse(asm_str)
            op = self.m.operation

            workgroup_ids = [0, 0]
            thread_ids = [63, 0, 0]
            sym_table = {}

            def get_dtype(dtype):
                if type(dtype) == F32Type:
                    return torch.float32
                if type(dtype) == F16Type:
                    return torch.float16

            def create_tensor(shape: list[int], dtype, value) -> torch.Tensor:
                if type(dtype) == F32Type or type(dtype) == F16Type:
                    value = float(value)
                return torch.ones(*shape, dtype=get_dtype(dtype)) * value

            def walk_operations(op, callback):
                for i in range(len(op.regions)):
                    region = op.regions[i]
                    for j in range(len(region.blocks)):
                        block = region.blocks[j]
                        for k in range(len(block.operations)):
                            child_op = block.operations[k]
                            callback(child_op)
                            walk_operations(child_op, callback)

            def callback(op):
                if not hasattr(callback, "for_op"):
                    callback.forOp = None
                if (
                    op.operation.parent.name == "func.func"
                    or op.operation.parent.name == "scf.for"
                ):
                    print("Op = ", op)
                    value = None
                    match type(op):
                        case arith_d.ConstantOp:
                            vtype = type(op.value.type)
                            if vtype == IndexType:
                                value = torch.Tensor([int(IntegerAttr(op.value))])
                            elif vtype == VectorType:
                                shape = op.value.type.shape
                                dtype = op.value.type.element_type
                                value = create_tensor(
                                    shape,
                                    dtype,
                                    op.attributes["value"].get_splat_value(),
                                )
                        case arith_d.MulIOp:
                            value = (
                                sym_table[op.operands[0]] * sym_table[op.operands[1]]
                            )
                        case arith_d.RemSIOp:
                            value = (
                                sym_table[op.operands[0]] % sym_table[op.operands[1]]
                            )
                        case arith_d.AddIOp:
                            value = (
                                sym_table[op.operands[0]] + sym_table[op.operands[1]]
                            )
                        case arith_d.SubIOp:
                            value = (
                                sym_table[op.operands[0]] - sym_table[op.operands[1]]
                            )
                        case arith_d.DivSIOp:
                            value = (
                                sym_table[op.operands[0]] // sym_table[op.operands[1]]
                            )
                        case amdgpu_d.LDSBarrierOp:
                            return
                        case amdgpu_d.MFMAOp:
                            lhs = sym_table[op.operands[0]]
                            rhs = sym_table[op.operands[1]]
                            acc = sym_table[op.operands[2]]
                            # TODO: Just use first row for now (which works for constant matrices)
                            # But figure out what to do in the general case
                            tmp = torch.outer(lhs, rhs)[0]
                            value = tmp + acc
                        case vector_d.LoadOp:
                            load_indices = []
                            for index in op.indices:
                                load_indices.append(sym_table[index])
                            memref = sym_table[op.base]
                            result_type = op.result.type
                            result_shape = result_type.shape
                            result_dtype = result_type.element_type
                            value = torch.zeros(
                                *result_shape, dtype=get_dtype(result_dtype)
                            )
                            # Row-major load
                            load_indices = [int(x) for x in load_indices]
                            print(load_indices)
                            for i in range(*result_shape):
                                value[i] = memref[
                                    int(load_indices[0]), int(load_indices[1] + i)
                                ]
                        case vector_d.ExtractStridedSliceOp:
                            vector = sym_table[op.vector]
                            offsets = []
                            sizes = []
                            for i in range(len(op.offsets)):
                                offsets.append(int(op.offsets[i]))
                            value = vector[offsets]
                        case vector_d.StoreOp:
                            store_indices = []
                            for index in op.indices:
                                store_indices.append(sym_table[index])
                            vector = sym_table[op.valueToStore]
                            memref = sym_table[op.base]
                            result_type = vector.type
                            result_shape = vector.shape
                            # Row-major load
                            store_indices = [int(x) for x in store_indices]
                            print(store_indices)
                            for i in range(*result_shape):
                                try:
                                    memref[
                                        int(store_indices[0]), int(store_indices[1] + i)
                                    ] = vector[i]
                                except:
                                    breakpoint()
                        case stream_d.DispatchWorkgroupIDOp:
                            index = int(op.attributes["dimension"])
                            value = workgroup_ids[index]
                            value = torch.Tensor([value])
                        case stream_d.BindingSubspanOp:
                            mtype = op.result.type
                            shape = mtype.shape
                            dtype = mtype.element_type
                            value = torch.ones(
                                shape, dtype=get_dtype(dtype)
                            ) * torch.randn((1,))
                        case gpu_d.ThreadIdOp:
                            dim = re.findall(r"^#gpu<dim (.*)>", str(op.dimension))[0]
                            if dim == "x":
                                value = thread_ids[0]
                            if dim == "y":
                                value = thread_ids[1]
                            if dim == "z":
                                value = thread_ids[2]
                            value = torch.Tensor([value])
                        case memref_d.AllocOp:
                            mtype = op.memref.type
                            shape = mtype.shape
                            dtype = mtype.element_type
                            value = torch.zeros(shape, dtype=get_dtype(dtype))
                        case scf_d.ForOp:
                            lb = int(sym_table[op.lowerBound])
                            ub = int(sym_table[op.upperBound])
                            callback.for_op = op
                            for init_arg, iter_arg in zip(
                                op.initArgs, op.inner_iter_args
                            ):
                                sym_table[iter_arg] = sym_table[init_arg]
                            for i in range(lb, ub):
                                print("i = ", i)
                                sym_table[op.induction_variable] = i
                                for k in range(len(op.body.operations)):
                                    callback(op.body.operations[k])
                            for result, iter_arg in zip(op.results, op.inner_iter_args):
                                sym_table[result] = sym_table[iter_arg]
                            return
                        case scf_d.YieldOp:
                            for result, iter_arg in zip(
                                op.operands, callback.for_op.inner_iter_args
                            ):
                                sym_table[iter_arg] = sym_table[result]
                            return
                        case func_d.ReturnOp:
                            return
                        case flow_d.DispatchOp:
                            return
                        case _:
                            breakpoint()

                    if type(op) != vector_d.StoreOp:
                        sym_table[op.result] = value

            walk_operations(op, callback)

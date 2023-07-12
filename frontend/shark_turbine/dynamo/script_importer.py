# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Imports from an FX graph by evaluating via torch.script."""

import functools
import io
from typing import List, Sequence

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch._functorch.compile_utils import strip_overloads
from torch.func import functionalize
import torch_mlir
from torch_mlir.dynamo import make_simple_dynamo_backend


class ScriptImporter:
    def __init__(self, text_mode: bool = False):
        self.text_mode = text_mode
        # Whether the output tuple was unwrapped to a single element.
        self.was_unwrapped: bool = False
        # Indices in the output that should be None (sorted).
        self.none_indices: list = []

    def __call__(self, fx_g: torch.fx.GraphModule, inputs: tuple) -> bytes:
        self._preprocess_fx(fx_g)
        ts_g = self._script_fx(fx_g, inputs)
        mlir_module = torch_mlir.compile(ts_g, inputs, output_type="linalg-on-tensors")
        # print(mlir_module)
        stream = io.BytesIO()
        if self.text_mode:
            mlir_module.operation.print(stream, binary=True, enable_debug_info=True)
        else:
            mlir_module.operation.write_bytecode(stream)
        return stream.getvalue()

    def _preprocess_fx(self, fx_g: torch.fx.GraphModule):
        self.none_indices = _remove_nones(fx_g)
        self.was_unwrapped = _unwrap_single_tuple_return(fx_g)

    def _script_fx(self, fx_g: torch.fx.GraphModule, inputs):
        gm = make_fx(
            functionalize(fx_g),
            decomposition_table=default_decompositions(),
        )(*inputs)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())
        gm.recompile()
        strip_overloads(gm)
        ts_g = torch.jit.script(gm)
        return ts_g


@functools.lru_cache
def default_decompositions():
    return get_decompositions(
        [
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
            torch.ops.aten.native_layer_norm,
            torch.ops.aten.masked_fill.Tensor,
            torch.ops.aten.masked_fill.Scalar,
        ]
    )


def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
    removed_indexes = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    curr_index = node_args_len - (i + 1)
                    if node_arg[curr_index] is None:
                        removed_indexes.append(curr_index)
                        node_arg.pop(curr_index)
                node.args = (tuple(node_arg),)
                break

    if len(removed_indexes) > 0:
        fx_g.graph.lint()
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    removed_indexes.sort()
    return removed_indexes


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    """
    Replace tuple with tuple element in functions that return one-element tuples.
    Returns true if an unwrapping took place, and false otherwise.
    """
    unwrapped_tuple = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    unwrapped_tuple = True
                    break

    if unwrapped_tuple:
        fx_g.graph.lint()
        fx_g.recompile()
    return unwrapped_tuple

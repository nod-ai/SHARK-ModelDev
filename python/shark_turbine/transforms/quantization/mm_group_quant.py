# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, cast

from iree.compiler.ir import (
    Operation,
    Type as IrType,
)

from ..rewriter import *
from ..builder import *


class TransposedMMResult(OpMatchResult):
    def __init__(
        self,
        op: Operation,
        *,
        weight_global: Operation,
        param_name: str,
        m: Optional[int],
        n: Optional[int],
        k: Optional[int],
        element_type: IrType,
    ):
        super().__init__(op)
        self.weight_global = weight_global
        self.param_name = param_name
        self.m = m
        self.n = n
        self.k = k
        self.element_type = element_type

    def __repr__(self):
        return f"TransposedMM(weight={self.param_name}, m={self.m}, n={self.n}, k={self.k}, element_type={self.element_type})"


class TransposedMMMatcher(NamedOpMatcher):
    def __init__(self, globals: GlobalsDict, builder: Builder):
        super().__init__("torch.aten.mm")
        self.globals = globals
        self.builder = builder

    def match(self, op: Operation):
        weight_transpose = Transpose2DMatcher()(op.operands[1])
        if not weight_transpose:
            return None
        weight_load = GlobalLoadMatcher(self.globals)(weight_transpose.input)
        if not weight_load or not weight_load.resolved_global:
            return None

        m, n = self.builder.get_tensor_dims(op.operands[0].type)
        _, k = self.builder.get_tensor_dims(op.operands[1].type)
        return TransposedMMResult(
            op,
            weight_global=weight_load.resolved_global,
            param_name=weight_load.global_ref,
            m=m,
            n=n,
            k=k,
            element_type=self.builder.get_tensor_element_type(op.operands[0].type),
        )


GROUP_MATMUL_TEMPLATE = r"""
module {{
  util.global private @{param_name}.quant {{noinline}} : tensor<{k}x{n_div}xi8>
  util.global private @{param_name}.quant.scale {{noinline}} : tensor<{k}x{group0}x{element_type}>
  util.global private @{param_name}.quant.zero_point {{noinline}} : tensor<{k}x{group0}x{element_type}>

  func.func private @{function_name}(%a : tensor<{m}x{n}x{element_type}>) -> tensor<{m}x{k}x{element_type}> {{
    %c0 = arith.constant 0 : index        
    %weight_raw = util.global.load @{param_name}.quant : tensor<{k}x{n_div}xi8>
    %m = tensor.dim %a, %c0 : tensor<{m}x{n}x{element_type}>
    %k = tensor.dim %weight_raw, %c0 : tensor<{k}x{n_div}xi8>
    %scale = util.global.load @{param_name}.quant.scale : tensor<{k}x{group0}x{element_type}>
    %zp = util.global.load @{param_name}.quant.zero_point : tensor<{k}x{group0}x{element_type}>
    %weight = flow.tensor.bitcast %weight_raw : tensor<{k}x{n_div}xi8> -> tensor<{k}x{n}x{lowp_type}>
    %a_exp = tensor.expand_shape %a [[0], [1, 2]] : tensor<{m}x{n}x{element_type}> into tensor<{m}x{group0}x{group1}x{element_type}>
    %weight_exp = tensor.expand_shape %weight [[0], [1, 2]] : tensor<{k}x{n}x{lowp_type}> into tensor<{k}x{group0}x{group1}x{lowp_type}>
    %empty_0 = tensor.empty() : tensor<{k}x{group0}x{group1}x{element_type}>
    %weight_cast = linalg.generic {{
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>, 
            affine_map<(d0, d1, d2) -> (d0, d1)>, 
            affine_map<(d0, d1, d2) -> (d0, d1)>, 
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
        iterator_types = ["parallel", "parallel", "parallel"] }} 
        ins(%weight_exp, %scale, %zp : tensor<{k}x{group0}x{group1}x{lowp_type}>, tensor<{k}x{group0}x{element_type}>, tensor<{k}x{group0}x{element_type}>) 
        outs(%empty_0 : tensor<{k}x{group0}x{group1}x{element_type}>) {{
    ^bb0(%in: {lowp_type}, %in_1: {element_type}, %in_2: {element_type}, %out: {element_type}):
        %16 = arith.extui %in : {lowp_type} to i32
        %17 = arith.uitofp %16 : i32 to {element_type}
        %18 = arith.subf %17, %in_2 : {element_type}
        %19 = arith.mulf %18, %in_1 : {element_type}
        linalg.yield %19 : {element_type}
    }} -> tensor<{k}x{group0}x{group1}x{element_type}>
    %cst = arith.constant 0.000000e+00 : {element_type}
    %empty_1_dyn = tensor.empty(%m, %k) : tensor<?x?x{element_type}>
    %empty_1 = tensor.cast %empty_1_dyn : tensor<?x?x{element_type}> to tensor<{m}x{k}x{element_type}>
    %zero_init = linalg.fill ins(%cst : {element_type}) outs(%empty_1 : tensor<{m}x{k}x{element_type}>) -> tensor<{m}x{k}x{element_type}>
    %result = linalg.generic {{
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, 
            affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, 
            affine_map<(d0, d1, d2, d3) -> (d0, d1)>], 
        iterator_types = ["parallel", "parallel", "reduction", "reduction"] }} 
        ins(%a_exp, %weight_cast : tensor<{m}x{group0}x{group1}x{element_type}>, tensor<{k}x{group0}x{group1}x{element_type}>) 
        outs(%zero_init : tensor<{m}x{k}x{element_type}>) {{
    ^bb0(%in: {element_type}, %in_1: {element_type}, %out: {element_type}):
        %16 = arith.mulf %in, %in_1 : {element_type}
        %17 = arith.addf %16, %out : {element_type}
        linalg.yield %17 : {element_type}
    }} -> tensor<{m}x{k}x{element_type}>
    return %result : tensor<{m}x{k}x{element_type}>
  }}
}}
"""


class MMGroupQuantRewriter(Pass):
    def run(self):
        globals = self.globals
        mms = match_children(self.funcs, TransposedMMMatcher(globals, self.builder))
        print(mms)

        for mr in mms:
            if mr.k is None or mr.n is None:
                continue
            self.rewrite(mr)

    def rewrite(self, mr: TransposedMMResult):
        none_to_q = lambda x: "?" if x is None else x
        function_name = f"compute__{mr.param_name}"
        inline_module_asm = GROUP_MATMUL_TEMPLATE.format(
            function_name=function_name,
            param_name=mr.param_name,
            lowp_type="i4",
            m=none_to_q(mr.m),
            n=none_to_q(mr.n),
            k=none_to_q(mr.k),
            n_div=mr.n // 2,
            group0=32,
            group1=128,
            element_type=mr.element_type,
        )
        print(inline_module_asm)
        inline_module = Operation.parse(inline_module_asm)
        print(inline_module)


if __name__ == "__main__":
    pass_main(MMGroupQuantRewriter)

# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "mmtfp",
]


@CustomOp.register(library=LIBRARY)
class mmtfp(CustomOp):
    """Performs a floating point matmul of an 'a' and transposed 'b' tensor.

    The types need not match: the bT tensor will be cast to the dtype of the
    'a' tensor.
    """

    signature = "mmtfp(Tensor a, Tensor bT) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape m, k
        bT_desc = ksel.arg_tensor(1)  # Shape n, k
        *batch_dims, a_m, a_k = a_desc.t.shape
        bT_n, bT_k, *rest = bT_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point and bT_desc.t.dtype.is_floating_point,
            lambda: f"mmtfp: Expected floating point",
        )
        torch._check(
            len(batch_dims) <= 1,
            lambda: f"mmtfp arg 'a': Expected 2d or 3d tensor (got {a_desc.t.shape})",
        )
        torch._check(
            not rest, f"mmtfp arg 'bT': Expected 2d tensor (got {bT_desc.t.shape})"
        )
        torch._check(
            a_k == bT_k,
            f"mmtfp arg 'bT': Expected matching K dimension ({a_k} vs {bT_k})",
        )

        # Specialize on the k and n dims.
        a_desc.specialize_dims(-1)
        bT_desc.specialize_all_dims()

        # Result 0: Shape batch..., m, n
        ksel.return_new_tensor(
            batch_dims + [a_m, bT_n], a_desc.t.dtype
        ).specialize_dims(-1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        bT = kb.arg_value(1)
        a_tensor_type = RankedTensorType(a.type)
        rank = a_tensor_type.rank
        bT_tensor_type = RankedTensorType(bT.type)
        n, k = bT_tensor_type.shape
        a_type_str = str(a_tensor_type.element_type)
        bT_type_str = str(bT_tensor_type.element_type)
        kwargs = {
            "n": n,
            "k": k,
            "a_type": a_type_str,
            "bT_type": bT_type_str,
        }
        if rank == 2:
            template_file = "mmtfp_2d.mlir"
            target_function_name = (
                f"turbine_llm_mmtfp_2d_{n}_{k}_{a_type_str}{bT_type_str}"
            )
        elif rank == 3:
            template_file = "mmtfp_3d.mlir"
            target_function_name = (
                f"turbine_llm_mmtfp_3d_{n}_{k}_{a_type_str}{bT_type_str}"
            )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            **kwargs,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))

# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "mmt_block_scaled_q8",
]


@CustomOp.register(library=LIBRARY)
class mmt_block_scaled_q8(CustomOp):
    """Generic block scaled matmul with transposed RHS.

    This corresponds to the BlockScaledLayout and operates on planar `d`
    and `qs` tensors as specified there:

    * `d`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`

    The LHS is expected to be a 3d tensor of shape [B, M, K]. The kernel
    will be specialized for all values of N, K and LHS dtype.
    """

    signature = "mmt_block_scaled_q8(Tensor a, Tensor d, Tensor qs) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape [b, ] m, k
        qs_desc = ksel.arg_tensor(2)  # Shape [N, K // BLOCK_SIZE, BLOCK_SIZE]
        d_desc = ksel.arg_tensor(1)  # Shape [N, K // BLOCK_SIZE, 1]

        # a arg
        *batch_dims, a_m, a_k = a_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point,
            lambda: f"mmt_block_scaled_q8 arg 'a': Expected floating point (got {a_desc.t.dtype})",
        )
        torch._check(
            len(batch_dims) == 1,
            lambda: f"mmt_block_scaled_q8 arg 'a': Expected 3d tensor (got {a_desc.t.shape})",
        )

        # qs arg
        qs_n, qs_group0, qs_bs, *rest = qs_desc.t.shape
        torch._check(
            len(rest) == 0 and (qs_group0 * qs_bs) == a_k,
            lambda: f"mmt_block_scaled_q8 arg 'qs': Incorrect shape (got {qs_desc.t.shape})",
        )

        # d arg
        d_n, d_group0, d_one, *rest = d_desc.t.shape
        torch._check(
            len(rest) == 0 and (d_group0 * qs_bs) == a_k and d_one == 1 and d_n == qs_n,
            lambda: f"mmt_block_scaled_q8 arg 'd': Incorrect shape (got {d_desc.t.shape})",
        )

        # Specialize on K, N, BS
        a_desc.specialize_dims(-1)
        qs_desc.specialize_all_dims()
        d_desc.specialize_all_dims()

        # Shape batch..., m, n
        c_desc = ksel.return_new_tensor(batch_dims + [a_m, d_n], dtype=a_desc.t.dtype)
        c_desc.specialize_dims(-1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        a_tensor_type = RankedTensorType(a.type)
        d = kb.arg_value(1)
        d_tensor_type = RankedTensorType(d.type)
        qs = kb.arg_value(2)
        qs_tensor_type = RankedTensorType(qs.type)

        rank = a_tensor_type.rank
        k = a_tensor_type.get_dim_size(rank - 1)
        n, group0, bs = qs_tensor_type.shape
        a_type_str = str(a_tensor_type.element_type)
        scale_type_str = str(d_tensor_type.element_type)

        template_file = "mmt_block_scaled_q8_3d.mlir"
        target_function_name = (
            f"turbine_llm_mmt_block_scaled_q8_3d_{n}_{k}_{bs}_{a_type_str}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            bs=bs,
            group0=group0,
            a_type=a_type_str,
            scale_type=scale_type_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
        print(kb.module_body.owner)

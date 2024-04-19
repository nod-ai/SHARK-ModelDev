# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "mmt_super_block_scaled_offset_q4_unsigned",
]


@CustomOp.register(library=LIBRARY)
class mmt_super_block_scaled_offset_q4_unsigned(CustomOp):
    """Super block scaled q4 matmul with transposed RHS.

    Arguments:

    * `a`: [B, M, K]
    * `d`: [N, SUP_COUNT, 1]
    * `dmin`: [N, SUP_COUNT, 1]
    * `sb_scales_hi`: [N, SUP_COUNT, SUB_COUNT // 4]
    * `sb_scales_lo`: [N, SUP_COUNT, SUB_COUNT // 2]
    * `sb_min_hi`: [N, SUP_COUNT, SUB_COUNT // 4]
    * `sb_mins_lo`: [N, SUP_COUNT, SUB_COUNT // 2]
    * `qs`: [N, SUP_COUNT, SUB_COUNT, BS // 2]

    Where: `K == SUP_COUNT * SUB_COUNT * BS`

    Given this and hi/lo combined into a single value, the dequantization
    formula is:

    ```
    d_scaled = (d * sb_scales).unsqueeze(-1)
    dmin_scaled = (dmin * sb_mins).unsqueeze(-1)
    return d_scaled * qs - dmin_scaled
    ```
    """

    signature = (
        "mmt_super_block_scaled_offset_q4_unsigned("
        "Tensor a, Tensor d, Tensor dmin, "
        "Tensor sb_scales_hi, Tensor sb_scales_low, "
        "Tensor sb_mins_hi, Tensor sb_mins_low, "
        "Tensor qs"
        ") -> (Tensor)"
    )

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)
        d_desc = ksel.arg_tensor(1)
        dmin_desc = ksel.arg_tensor(2)
        sb_scales_hi_desc = ksel.arg_tensor(3)
        sb_scales_low_desc = ksel.arg_tensor(4)
        sb_mins_hi_desc = ksel.arg_tensor(5)
        sb_mins_low_desc = ksel.arg_tensor(6)
        qs_desc = ksel.arg_tensor(7)

        # a arg
        *batch_dims, m, k = a_desc.t.shape
        a_desc.specialize_dims(-1)
        torch._check(
            a_desc.t.dtype.is_floating_point,
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'a': Expected floating point (got {a_desc.t.dtype})",
        )
        torch._check(
            len(batch_dims) == 1,
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'a': Expected 3d tensor (got {a_desc.t.shape})",
        )

        # qs arg
        n, sup_count, sub_count, bs_div2 = qs_desc.t.shape
        qs_desc.specialize_all_dims()
        bs = bs_div2 * 2
        torch._check(
            k == (sup_count * sub_count * bs),
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'qs': Incorrect shape (got {qs_desc.t.shape}, k={k})",
        )

        # d arg
        v_n, v_sup_count, one = d_desc.t.shape
        d_desc.specialize_all_dims()
        torch._check(
            v_n == n and v_sup_count == sup_count and one == 1,
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'd': Incorrect shape (got {d_desc.t.shape})",
        )

        # dmin arg
        v_n, v_sup_count, one = dmin_desc.t.shape
        dmin_desc.specialize_all_dims()
        torch._check(
            v_n == n and v_sup_count == sup_count and one == 1,
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'dmin': Incorrect shape (got {d_desc.t.shape})",
        )

        # sb_scales_hi arg
        v_n, v_sup_count, v_sub_div4 = sb_scales_hi_desc.t.shape
        sb_scales_hi_desc.specialize_all_dims()
        torch._check(
            v_n == n and v_sup_count == sup_count and v_sub_div4 == (sub_count // 4),
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'sb_scales_hi': Incorrect shape (got {sb_scales_hi_desc.t.shape})",
        )

        # sb_scales_low arg
        v_n, v_sup_count, v_sub_div2 = sb_scales_low_desc.t.shape
        sb_scales_low_desc.specialize_all_dims()
        torch._check(
            v_n == n and v_sup_count == sup_count and v_sub_div2 == (sub_count // 2),
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'sb_scales_low': Incorrect shape (got {sb_scales_low_desc.t.shape})",
        )

        # sb_mins_hi arg
        v_n, v_sup_count, v_sub_div4 = sb_mins_hi_desc.t.shape
        sb_mins_hi_desc.specialize_all_dims()
        torch._check(
            v_n == n and v_sup_count == sup_count and v_sub_div4 == (sub_count // 4),
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'sb_mins_hi': Incorrect shape (got {sb_mins_hi_desc.t.shape})",
        )

        # sb_mins_low arg
        v_n, v_sup_count, v_sub_div2 = sb_mins_low_desc.t.shape
        sb_mins_low_desc.specialize_all_dims()
        torch._check(
            v_n == n and v_sup_count == sup_count and v_sub_div2 == (sub_count // 2),
            lambda: f"mmt_super_block_scaled_offset_q4_unsigned arg 'sb_mins_low': Incorrect shape (got {sb_mins_low_desc.t.shape})",
        )

        # c return
        # Shape batch..., m, n
        c_desc = ksel.return_new_tensor(batch_dims + [m, n], dtype=a_desc.t.dtype)
        c_desc.specialize_dims(-1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        a_tensor_type = RankedTensorType(a.type)
        *_, k = a_tensor_type.shape
        d = kb.arg_value(1)
        d_tensor_type = RankedTensorType(d.type)
        qs = kb.arg_value(7)
        qs_tensor_type = RankedTensorType(qs.type)
        n, sup_count, sub_count, bs_div2 = qs_tensor_type.shape
        bs = bs_div2 * 2
        a_type_str = str(a_tensor_type.element_type)
        scale_type_str = str(d_tensor_type.element_type)

        template_file = "mmt_super_block_scaled_offset_q4_unsigned_3d.mlir"
        target_function_name = f"mmt_super_block_scaled_offset_q4_unsigned_3d_{n}_{k}_{sup_count}_{sub_count}_{bs}_{a_type_str}"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            sup_count=sup_count,
            sub_count=sub_count,
            sub_div4=sub_count // 4,
            sub_div2=sub_count // 2,
            bs=bs,
            bs_div2=bs_div2,
            a_type=a_type_str,
            scale_type=scale_type_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
        print(kb.module_body.owner)

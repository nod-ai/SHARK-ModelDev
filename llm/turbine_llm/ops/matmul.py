# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "mmtfp",
    "mmt_block_scaled_offset_q4_unsigned",
    "mmt_block_scaled_q8",
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
        *batch_dims, a_m, a_k = a_desc.t.shape
        if not a_desc.t.dtype.is_floating_point:
            raise ValueError(
                f"mmtfp arg 'a': Expected floating point (got {a_desc.t.dtype})"
            )
        if len(batch_dims) > 1:
            raise ValueError(
                f"mmtfp arg 'a': Expected 2d or 3d tensor (got {a_desc.t.shape})"
            )
        bT_desc = ksel.arg_tensor(1)  # Shape n, k
        bT_n, bT_k, *rest = bT_desc.t.shape
        if not a_desc.t.dtype.is_floating_point:
            raise ValueError(
                f"mmtfp arg 'bT': Expected floating pount (got {bT_desc.t.dtype})"
            )
        if rest:
            raise ValueError(
                f"mmtfp arg 'bT': Expected 2d tensor (got {bT_desc.t.shape})"
            )
        if a_k != bT_k:
            raise ValueError(
                f"mmtfp arg 'bT': Expected matching K dimension ({a_k} vs {bT_k})"
            )

        # Specialize on the k and n dims.
        a_desc.spec_dims[-1] = bT_k
        bT_desc.spec_dims[0] = bT_n
        bT_desc.spec_dims[1] = bT_k

        c = torch.empty(batch_dims + [a_m, bT_n], dtype=a_desc.t.dtype)
        c_desc = ksel.return_tensor(c)  # Shape batch..., m, n
        c_desc.spec_dims[-1] = bT_n

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
        # a arg
        a_desc = ksel.arg_tensor(0)  # Shape [b, ] m, k
        *batch_dims, a_m, a_k = a_desc.t.shape
        if not a_desc.t.dtype.is_floating_point:
            raise ValueError(
                f"mmt_block_scaled_q8 arg 'a': Expected floating point (got {a_desc.t.dtype})"
            )
        if len(batch_dims) != 1:
            raise ValueError(
                f"mmt_block_scaled_q8 arg 'a': Expected 3d tensor (got {a_desc.t.shape})"
            )

        # qs arg
        qs_desc = ksel.arg_tensor(2)  # Shape [N, K // BLOCK_SIZE, BLOCK_SIZE]
        qs_n, qs_group0, qs_bs, *rest = qs_desc.t.shape
        if rest or (qs_group0 * qs_bs) != a_k:
            raise ValueError(
                f"mmt_block_scaled_q8 arg 'qs': Incorrect shape (got {qs_desc.t.shape})"
            )

        # d arg
        d_desc = ksel.arg_tensor(1)  # Shape [N, K // BLOCK_SIZE, 1]
        d_n, d_group0, d_one, *rest = d_desc.t.shape
        if rest or (d_group0 * qs_bs) != a_k or d_one != 1 or d_n != qs_n:
            raise ValueError(
                f"mmt_block_scaled_q8 arg 'd': Incorrect shape (got {d_desc.t.shape})"
            )

        # Specialize on K, N, BS
        a_desc.spec_dims[-1] = a_k
        qs_desc.spec_dims[:] = qs_desc.t.shape
        d_desc.spec_dims[:] = d_desc.t.shape

        c = torch.empty(batch_dims + [a_m, d_n], dtype=a_desc.t.dtype)
        c_desc = ksel.return_tensor(c)  # Shape batch..., m, n
        c_desc.spec_dims[-1] = d_n

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


@CustomOp.register(library=LIBRARY)
class mmt_block_scaled_offset_q4_unsigned(CustomOp):
    """Generic block scaled matmul with transposed RHS.

    This corresponds to the BlockScaledLayout and operates on planar `d`
    and `qs` tensors as specified there:

    * `d`: `[N, K // BLOCK_SIZE, 1]`
    * `qs`: `[N, K // BLOCK_SIZE, BLOCK_SIZE // 2]` (of uint8)
    * `m`: `[N, K // BLOCK_SIZE, 1]`

    The LHS is expected to be a 3d tensor of shape [B, M, K]. The kernel
    will be specialized for all values of N, K and LHS dtype.
    """

    signature = "mmt_block_scaled_offset_q4_unsigned(Tensor a, Tensor d, Tensor qs, Tensor m) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        # a arg
        a_desc = ksel.arg_tensor(0)  # Shape [b, ] m, k
        *batch_dims, a_m, a_k = a_desc.t.shape
        if not a_desc.t.dtype.is_floating_point:
            raise ValueError(
                f"mmt_block_scaled_offset_q4_unsigned arg 'a': Expected floating point (got {a_desc.t.dtype})"
            )
        if len(batch_dims) != 1:
            raise ValueError(
                f"mmt_block_scaled_offset_q4_unsigned arg 'a': Expected 3d tensor (got {a_desc.t.shape})"
            )

        # qs arg
        qs_desc = ksel.arg_tensor(2)  # Shape [N, K // BLOCK_SIZE, BLOCK_SIZE // 2]
        qs_n, qs_group0, qs_bs_div_2, *rest = qs_desc.t.shape
        if rest or (qs_group0 * qs_bs_div_2 * 2) != a_k:
            raise ValueError(
                f"mmt_block_scaled_offset_q4_unsigned arg 'qs': Incorrect shape (got {qs_desc.t.shape})"
            )
        block_size = qs_bs_div_2 * 2

        # d arg
        d_desc = ksel.arg_tensor(1)  # Shape [N, K // BLOCK_SIZE, 1]
        d_n, d_group0, d_one, *rest = d_desc.t.shape
        if rest or (d_group0 * block_size) != a_k or d_one != 1 or d_n != qs_n:
            raise ValueError(
                f"mmt_block_scaled_offset_q4_unsigned arg 'd': Incorrect shape (got {d_desc.t.shape})"
            )

        # m arg
        m_desc = ksel.arg_tensor(3)  # Shape [N, K // BLOCK_SIZE, 1]
        m_n, m_group0, m_one, *rest = m_desc.t.shape
        if rest or (m_group0 * block_size) != a_k or m_one != 1 or m_n != qs_n:
            raise ValueError(
                f"mmt_block_scaled_offset_q4_unsigned arg 'm': Incorrect shape (got {m_desc.t.shape})"
            )
        if m_desc.t.dtype != d_desc.t.dtype:
            raise ValueError(
                f"mmt_block_scaled_offset_q4_unsigned arg 'm': Incorrect dtype (got {m_desc.t.dtype})"
            )

        # Specialize on K, N, BS
        a_desc.spec_dims[-1] = a_k
        qs_desc.spec_dims[:] = qs_desc.t.shape
        d_desc.spec_dims[:] = d_desc.t.shape
        m_desc.spec_dims[:] = m_desc.t.shape

        c = torch.empty(batch_dims + [a_m, d_n], dtype=a_desc.t.dtype)
        c_desc = ksel.return_tensor(c)  # Shape batch..., m, n
        c_desc.spec_dims[-1] = d_n

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        a_tensor_type = RankedTensorType(a.type)
        d = kb.arg_value(1)
        d_tensor_type = RankedTensorType(d.type)
        qs = kb.arg_value(2)
        qs_tensor_type = RankedTensorType(qs.type)

        rank = a_tensor_type.rank
        k = a_tensor_type.get_dim_size(rank - 1)
        n, group0, bs_i8 = qs_tensor_type.shape
        bs = bs_i8 * 2  # 2 nibbles per byte.
        a_type_str = str(a_tensor_type.element_type)
        scale_type_str = str(d_tensor_type.element_type)

        template_file = "mmt_block_scaled_offset_q4_unsigned.mlir"
        target_function_name = f"turbine_llm_mmt_block_scaled_offset_q4_unsigned_3d_{n}_{k}_{bs}_{a_type_str}"

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            n=n,
            k=k,
            bs=bs,
            bs_i8=bs_i8,
            group0=group0,
            a_type=a_type_str,
            scale_type=scale_type_str,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
        print(kb.module_body.owner)

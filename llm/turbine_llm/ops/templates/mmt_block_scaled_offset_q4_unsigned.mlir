// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!lowp_type = i4
!a_type = {a_type}
!scale_type = {scale_type}
!a_tensor_type = tensor<?x?x{k}x!a_type>
!aexp_tensor_type = tensor<?x?x{group0}x{bs}x!a_type>
!qs_raw_tensor_type = tensor<{n}x{group0}x{bs_i8}xi8>
!qs_tensor_type = tensor<{n}x{group0}x{bs}x!lowp_type>
!d_tensor_type = tensor<{n}x{group0}x1x!scale_type>
!m_tensor_type = tensor<{n}x{group0}x1x!scale_type>
// TODO: We should really have an accumulator type, which will be needed for
// f16 and below.
!c_tensor_type = tensor<?x?x{n}x!a_type>
!b_grouped_tensor_type = tensor<{n}x{group0}x{bs}x!a_type>

module {{

util.func private @turbine_llm_mmt_block_scaled_offset_q4_unsigned_3d_{n}_{k}_{bs}_{a_type}(
    %a: !a_tensor_type, %d: !d_tensor_type, %qs_raw: !qs_raw_tensor_type, %m: !m_tensor_type)
    -> !c_tensor_type {{
  %zero = arith.constant 0.0: !a_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch0_dim = tensor.dim %a, %c0 : !a_tensor_type
  %m_dim = tensor.dim %a, %c1 : !a_tensor_type

  // Cast qs_raw from i8 to lowp type.
  %qs = flow.tensor.bitcast %qs_raw : !qs_raw_tensor_type -> !qs_tensor_type

  // Dequantize.
  %b_grouped = tensor.empty() : !b_grouped_tensor_type
  %b_grouped_dequant = linalg.generic {{
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
          affine_map<(d0, d1, d2) -> (d0, d1, 0)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>], 
      iterator_types = ["parallel", "parallel", "parallel"] }} 
      ins(%d, %m, %qs : !d_tensor_type, !m_tensor_type, !qs_tensor_type)
      outs(%b_grouped : !b_grouped_tensor_type) {{
  ^bb0(%d_element: !scale_type, %m_element: !scale_type, %q_element: !lowp_type, %out: !a_type):
      %q_element_ext = arith.extui %q_element : !lowp_type to i32
      %q_element_fp = arith.uitofp %q_element_ext : i32 to !a_type
      %d_element_ext = arith.extf %d_element : !scale_type to !a_type
      %m_element_ext = arith.extf %m_element : !scale_type to !a_type
      %q_element_scaled = arith.mulf %q_element_fp, %d_element_ext : !a_type
      %q_element_offset = arith.addf %q_element_scaled, %m_element_ext : !a_type
      linalg.yield %q_element_offset : !a_type
  }} -> !b_grouped_tensor_type

  // Expand %a to have the same blocked reduction structure.
  %aexp = tensor.expand_shape %a [[0], [1], [2, 3]] : !a_tensor_type into !aexp_tensor_type

  // Grouped, batch mm.
  %result_empty = tensor.empty(%batch0_dim, %m_dim) : !c_tensor_type
  %result_fill = linalg.fill ins(%zero: !a_type) outs(%result_empty: !c_tensor_type) -> !c_tensor_type
  %result = linalg.generic {{
      indexing_maps = [
          // d0 = b, d1 = m, d2 = n, d3 = group0 (r), d4 = block (r)
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>, 
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], 
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"] }} 
      ins(%aexp, %b_grouped_dequant : !aexp_tensor_type,  !b_grouped_tensor_type)
      outs(%result_fill : !c_tensor_type) {{
  ^bb0(%a_element: !a_type, %b_element: !a_type, %out: !a_type):
      %bmm_mul = arith.mulf %a_element, %b_element : !a_type
      %bmm_accum = arith.addf %bmm_mul, %out : !a_type
      linalg.yield %bmm_accum : !a_type
  }} -> !c_tensor_type

  util.return %result : !c_tensor_type
}}

}}

// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!a_type = {a_type}
!scale_type = {scale_type}
!a_tensor_type = tensor<?x?x{k}x!a_type>
!aexp_tensor_type = tensor<?x?x{sup_count}x{sub_count}x{bs}x!a_type>
!qs_tensor_type = tensor<{n}x{sup_count}x{sub_count}x{bs}xi4>
!qs_tensor_i8_type = tensor<{n}x{sup_count}x{sub_count}x{bs_div2}xi8>
!d_tensor_type = tensor<{n}x{sup_count}x1x!scale_type>
!dmin_tensor_type = tensor<{n}x{sup_count}x1x!scale_type>
!sb_hi_i8_type = tensor<{n}x{sup_count}x{sub_div4}xi8>
!sb_low_i8_type = tensor<{n}x{sup_count}x{sub_div2}xi8>
!sb_hi_i2_type = tensor<{n}x{sup_count}x{sub_count}xi2>
!sb_low_i4_type = tensor<{n}x{sup_count}x{sub_count}xi4>

// TODO: We should really have an accumulator type, which will be needed for
// f16 and below.
!c_tensor_type = tensor<?x?x{n}x!a_type>
!b_grouped_tensor_type = tensor<{n}x{sup_count}x{sub_count}x{bs}x!a_type>

module {{

util.func private @mmt_super_block_scaled_offset_q4_unsigned_3d_{n}_{k}_{sup_count}_{sub_count}_{bs}_{a_type}(
    %a: !a_tensor_type, 
    %d: !d_tensor_type, 
    %dmin: !dmin_tensor_type,
    %sb_scales_hi_i8: !sb_hi_i8_type,
    %sb_scales_low_i8: !sb_low_i8_type,
    %sb_mins_hi_i8: !sb_hi_i8_type,
    %sb_mins_low_i8: !sb_low_i8_type,
    %qs_i8: !qs_tensor_i8_type)
    -> !c_tensor_type {{
  %zero = arith.constant 0.0: !a_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %batch0_dim = tensor.dim %a, %c0 : !a_tensor_type
  %m_dim = tensor.dim %a, %c1 : !a_tensor_type

  // Bitcast from i8 to i4/i2 packings.
  %qs = flow.tensor.bitcast %qs_i8 : !qs_tensor_i8_type -> !qs_tensor_type
  %sb_scales_hi = flow.tensor.bitcast %sb_scales_hi_i8 : !sb_hi_i8_type -> !sb_hi_i2_type
  %sb_scales_low = flow.tensor.bitcast %sb_scales_low_i8 : !sb_low_i8_type -> !sb_low_i4_type
  %sb_mins_hi = flow.tensor.bitcast %sb_mins_hi_i8 : !sb_hi_i8_type -> !sb_hi_i2_type
  %sb_mins_low = flow.tensor.bitcast %sb_mins_low_i8 : !sb_low_i8_type -> !sb_low_i4_type

  // Dequantize.
  %b_grouped = tensor.empty() : !b_grouped_tensor_type
  %b_grouped_dequant = linalg.generic {{
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,  // qs[n, sup, sub, bs]
          affine_map<(d0, d1, d2, d3) -> (d0, d1, 0)>,       // d[n, sup, 1]
          affine_map<(d0, d1, d2, d3) -> (d0, d1, 0)>,       // dmin[n, sup, 1]
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,      // sb_scales_hi[n, sup, sub]
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,      // sb_scales_low[n, sup, sub]
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,      // sb_mins_hi[n, sup, sub]
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,      // sb_mins_low[n, sup, sub]
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>   // out b_grouped[n, sup, sub, bs]
      ], 
      iterator_types = ["parallel", "parallel", "parallel", "parallel"] }}
      ins(
        %qs, %d, %dmin, %sb_scales_hi, %sb_scales_low, %sb_mins_hi, %sb_mins_low :
        !qs_tensor_type, !d_tensor_type, !dmin_tensor_type, 
        !sb_hi_i2_type, !sb_low_i4_type, !sb_hi_i2_type, !sb_low_i4_type
      )
      outs(%b_grouped : !b_grouped_tensor_type) {{
  ^bb0(%q_element: i4, %d_element: !scale_type, %dmin_element: !scale_type,
       %sb_scales_hi_element: i2, %sb_scales_low_element: i4,
       %sb_mins_hi_element: i2, %sb_mins_low_element: i4,
        %out: !a_type):
      %shift_4 = arith.constant 4 : i32
      %d_element_ext = arith.extf %d_element : !scale_type to !a_type
      %dmin_element_ext = arith.extf %dmin_element : !scale_type to !a_type
      
      // Combine sub-block scale.
      %sb_scale_low_i32 = arith.extui %sb_scales_low_element : i4 to i32
      %sb_scale_hi_i32 = arith.extui %sb_scales_hi_element : i2 to i32
      %sb_scale_hi_i32_shift = arith.shli %sb_scale_hi_i32, %shift_4 : i32
      %sb_scale_i32 = arith.ori %sb_scale_low_i32, %sb_scale_hi_i32_shift : i32
      %sb_scale_float = arith.uitofp %sb_scale_i32 : i32 to !a_type

      // Combine sub-block min.
      %sb_min_low_i32 = arith.extui %sb_mins_low_element : i4 to i32
      %sb_min_hi_i32 = arith.extui %sb_mins_hi_element : i2 to i32
      %sb_min_hi_i32_shift = arith.shli %sb_min_hi_i32, %shift_4 : i32
      %sb_min_i32 = arith.ori %sb_min_low_i32, %sb_min_hi_i32 : i32
      %sb_min_float = arith.uitofp %sb_min_i32 : i32 to !a_type

      // Dequant equation.
      %q_element_i32 = arith.extui %q_element : i4 to i32
      %q_element_ext = arith.uitofp %q_element_i32 : i32 to !a_type
      %d_scaled = arith.mulf %d_element_ext, %sb_scale_float : !a_type
      %dmin_scaled = arith.mulf %dmin_element_ext, %sb_min_float : !a_type
      %q_scaled = arith.mulf %d_scaled, %q_element_ext : !a_type
      %q_shifted = arith.subf %q_scaled, %dmin_scaled : !a_type
      linalg.yield %q_shifted : !a_type
  }} -> !b_grouped_tensor_type

  // Expand %a to have the same blocked reduction structure (sup, sub, block).
  %aexp = tensor.expand_shape %a [[0], [1], [2, 3, 4]] : !a_tensor_type into !aexp_tensor_type

  // Grouped, batch mm.
  %result_empty = tensor.empty(%batch0_dim, %m_dim) : !c_tensor_type
  %result_fill = linalg.fill ins(%zero: !a_type) outs(%result_empty: !c_tensor_type) -> !c_tensor_type
  %result = linalg.generic {{
      indexing_maps = [
          // d0 = b, d1 = m, d2 = n, d3 = sup (r), d4 = sub (r), d5 = block (r)
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d5)>,  // aexp
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>,      // b_grouped_dequant
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>       // out
      ], 
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"] }} 
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

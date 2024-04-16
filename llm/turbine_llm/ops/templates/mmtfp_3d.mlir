// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!a_type = {a_type}
!bT_type = {bT_type}
!a_tensor_type = tensor<?x?x{k}x!a_type>
!bT_tensor_type = tensor<{n}x{k}x!bT_type>
!c_tensor_type = tensor<?x?x{n}x!a_type>
!bT_broadcast_tensor_type = tensor<?x{n}x{k}x!bT_type>

module {{

util.func private @turbine_llm_mmtfp_3d_{n}_{k}_{a_type}{bT_type}(
    %a: !a_tensor_type, %bT: !bT_tensor_type)
    -> !c_tensor_type {{
  %zero = arith.constant 0.000000e+00 : !a_type
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b0 = tensor.dim %a, %c0 : !a_tensor_type
  %m = tensor.dim %a, %c1 : !a_tensor_type
  %bT_broadcast_empty = tensor.empty(%b0) : !bT_broadcast_tensor_type
  %bT_broadcast = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"] }}
    ins(%bT: !bT_tensor_type)
    outs(%bT_broadcast_empty: !bT_broadcast_tensor_type) {{
      ^bb0(%in: !bT_type, %out: !bT_type):
        linalg.yield %in : !bT_type
    }} -> !bT_broadcast_tensor_type
  %result_empty = tensor.empty(%b0, %m) : !c_tensor_type
  %result_init = linalg.fill 
    ins(%zero : !a_type) 
    outs(%result_empty: !c_tensor_type) -> !c_tensor_type
  %result = linalg.batch_matmul_transpose_b
    ins (%a, %bT_broadcast: !a_tensor_type, !bT_broadcast_tensor_type)
    outs (%result_init: !c_tensor_type) -> !c_tensor_type
  util.return %result : !c_tensor_type
}}

}}

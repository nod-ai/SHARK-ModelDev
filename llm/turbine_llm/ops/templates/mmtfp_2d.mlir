// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!a_type = {a_type}
!bT_type = {bT_type}
!a_tensor_type = tensor<?x{k}x!a_type>
!bT_tensor_type = tensor<{n}x{k}x!bT_type>
!c_tensor_type = tensor<?x{n}x!a_type>

module {{

util.func private @turbine_llm_mmtfp_2d_{n}_{k}_{a_type}{bT_type}(
    %a: !a_tensor_type, %bT: !bT_tensor_type)
    -> !c_tensor_type {{
  %zero = arith.constant 0.000000e+00 : {a_type}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %a, %c0 : !a_tensor_type
  %result_empty = tensor.empty(%m) : !c_tensor_type
  %result_init = linalg.fill 
    ins(%zero : {a_type}) 
    outs(%result_empty: !c_tensor_type) -> !c_tensor_type
  %result = linalg.matmul_transpose_b
    ins (%a, %bT: !a_tensor_type, !bT_tensor_type)
    outs (%result_init: !c_tensor_type) -> !c_tensor_type
  util.return %result : !c_tensor_type
}}

}}

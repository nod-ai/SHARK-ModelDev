// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module {{

util.func private @turbine_llm_mmtfp_2d_{n}_{k}_{a_type}{bT_type}(
    %a: tensor<?x{k}x{a_type}>, %bT: tensor<{n}x{k}x{bT_type}>)
    -> tensor<?x{n}x{a_type}> {{
  %zero = arith.constant 0.000000e+00 : {a_type}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %a, %c0 : tensor<?x{k}x{a_type}>
  %result_empty = tensor.empty(%m) : tensor<?x{n}x{a_type}>
  %result_init = linalg.fill 
    ins(%zero : {a_type}) 
    outs(%result_empty: tensor<?x{n}x{a_type}>) -> tensor<?x{n}x{a_type}>
  %result = linalg.matmul_transpose_b
    ins (%a, %bT: tensor<?x{k}x{a_type}>, tensor<{n}x{k}x{bT_type}>)
    outs (%result_init: tensor<?x{n}x{a_type}>) -> tensor<?x{n}x{a_type}>
  util.return %result : tensor<?x{n}x{a_type}>
}}

}}

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shark-turbine/compiler/InputConversion/Torch/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace TorchInput {

namespace {
#define GEN_PASS_REGISTRATION
#include "shark-turbine/compiler/InputConversion/Torch/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerTMTensorConversionPasses() {
  // Generated.
  registerPasses();
}

} // namespace TorchInput
} // namespace iree_compiler
} // namespace mlir

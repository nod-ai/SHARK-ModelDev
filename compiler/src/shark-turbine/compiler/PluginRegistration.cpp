// Copyright 2023 Nod Labs, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "shark-turbine/compiler/InputConversion/Torch/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace {

struct SharkTurbineOptions {
  void bindOptions(OptionsBinder &binder) {}
};

// The shark-turbine plugin provides dialects, passes and opt-in options.
// Therefore, it is appropriate for default activation.
struct SharkTurbineSession
    : public PluginSession<SharkTurbineSession, SharkTurbineOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    mlir::torch::registerTorchPasses();
    mlir::torch::registerTorchConversionPasses();
    mlir::torch::registerConversionPasses();
    TorchInput::registerTMTensorConversionPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<torch::Torch::TorchDialect>();
    registry.insert<torch::TorchConversion::TorchConversionDialect>();
    registry.insert<mlir::torch::TMTensor::TMTensorDialect>();
  }
};

} // namespace

} // namespace mlir::iree_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::SharkTurbineOptions);

extern "C" bool iree_register_compiler_plugin_shark_turbine(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::SharkTurbineSession>(
      "shark-turbine");
  return true;
}

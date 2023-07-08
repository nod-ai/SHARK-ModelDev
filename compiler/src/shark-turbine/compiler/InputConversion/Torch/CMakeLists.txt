iree_add_all_subdirs()

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    PassHeaders
  HDRS
    "PassDetail.h"
    "Passes.h"
    "Passes.h.inc"
  DEPS
    ::PassesIncGen
    MLIRPass
    MLIRTransforms
  PUBLIC
)

iree_cc_library(
  NAME
    TMTensor
  HDRS
    "Passes.h"
  SRCS
    "ConvertTMTensorToLinalgExt.cpp"
    "Passes.cpp"
  DEPS
    ::PassHeaders
    ::PassesIncGen
    IREELinalgExtDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRTensorDialect
    MLIRTransforms
    torch-mlir-dialects::TMTensorDialectIR
  PUBLIC
)

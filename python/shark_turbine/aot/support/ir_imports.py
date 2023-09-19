# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unifies all imports of iree.compiler.ir into one place."""

from iree.compiler.ir import (
    Block,
    BlockArgument,
    Context,
    DenseElementsAttr,
    FlatSymbolRefAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    Location,
    MLIRError,
    Module,
    OpResult,
    Operation,
    RankedTensorType,
    ShapedType,
    StringAttr,
    SymbolTable,
    Type as IrType,
    TypeAttr,
    UnitAttr,
    # Types.
    ComplexType,
    BF16Type,
    F16Type,
    F32Type,
    F64Type,
    IntegerType,
    RankedTensorType,
    Value,
)

from iree.compiler.passmanager import (
    PassManager,
)

from iree.compiler.dialects import (
    flow as flow_d,
    func as func_d,
    util as util_d,
    arith as arith_d,
    tensor as tensor_d,
)

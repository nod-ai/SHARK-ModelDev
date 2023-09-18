# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unifies all imports of iree.compiler.ir into one place."""

from iree.compiler.ir import (
    Context,
    DenseElementsAttr,
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    Location,
    MLIRError,
    Module,
    Operation,
    RankedTensorType,
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
    func as func_d,
    util as util_d,
)

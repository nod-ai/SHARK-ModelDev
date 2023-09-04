# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from iree.compiler.ir import (
    Context,
    InsertionPoint,
    Location,
    Operation,
    StringAttr,
    SymbolTable,
)

logger = logging.getLogger("shark_turbine.aot")


class ModuleBuilder:
    """Wrapper around module and IR accounting for a module being built."""

    def __init__(self, module_op: Operation):
        self.module_op = module_op
        self.context = module_op.context
        self.body = module_op.regions[0].blocks[0]
        self.symbol_table = SymbolTable(module_op)
        self.ip = InsertionPoint(self.body)

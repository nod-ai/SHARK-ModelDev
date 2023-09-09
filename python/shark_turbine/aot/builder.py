# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence, Tuple

import logging

from iree.compiler.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    Location,
    Operation,
    StringAttr,
    SymbolTable,
    Type,
)

from iree.compiler.dialects import (
    func as func_d,
)

from ..dynamo.importer import (
    ContextCache,
)

logger = logging.getLogger("shark_turbine.aot")


class ModuleBuilder:
    """Wrapper around module and IR accounting for a module being built."""

    __slots__ = [
        "body",
        "context",
        "module_op",
        "symbol_table",
        "ip",
        "cache",
    ]

    def __init__(self, module_op: Operation):
        self.module_op = module_op
        self.context = module_op.context
        self.body = module_op.regions[0].blocks[0]
        self.symbol_table = SymbolTable(module_op)
        self.ip = InsertionPoint(self.body)
        self.cache = ContextCache(self.context)

    def finalize_construct(self):
        self.module_op.verify()

    def create_func_op(
        self,
        symbol_name: str,
        argument_types: Sequence[Type],
        is_public: bool = True,
        add_entry_block: bool = True,
    ) -> Tuple[str, func_d.FuncOp]:
        with self.ip:
            ftype = FunctionType.get(argument_types, [])
            func_op = func_d.FuncOp(symbol_name, ftype)
            if not is_public:
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
            if add_entry_block:
                func_op.add_entry_block()
            self.symbol_table.insert(func_op)
            actual_symbol_name = StringAttr(func_op.attributes["sym_name"]).value
            return actual_symbol_name, func_op

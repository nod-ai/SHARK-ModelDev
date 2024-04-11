# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import logging
from pathlib import Path

from shark_turbine.support.ir_imports import (
    util_d,
    FlatSymbolRefAttr,
    FunctionType,
    Operation,
    RankedTensorType,
    StringAttr,
    TypeAttr,
    Value,
)

from shark_turbine.runtime.op_reg import (
    def_library,
    CustomOp,
    KernelBuilder,
    KernelSelection,
)

from shark_turbine.transforms.merger import Merger

from ..utils.logging import get_logger

LIBRARY = def_library("turbine_llm")
TEMPLATES_DIR = Path(__file__).parent / "templates"
logger = get_logger("turbine_llm.ops")


def call_function(target_function: Operation, *operands: Value) -> Sequence[Value]:
    target_symbol = FlatSymbolRefAttr.get(
        StringAttr(target_function.attributes["sym_name"]).value_bytes
    )
    ftype = FunctionType(TypeAttr(target_function.attributes["function_type"]).value)
    return Operation.create(
        "util.call",
        results=ftype.results,
        operands=operands,
        attributes={
            "callee": target_symbol,
        },
    ).results


def inline_template_function(
    kb: KernelBuilder, template_file: str, function_name: str, **kwargs
) -> Operation:
    """Inlines a template module by first expanding its ASM via **kwargs.

    Returns the inlined symbol `function_name`, which is expected to have been
    in the template.
    """
    try:
        return kb.symbol_table[function_name]
    except KeyError:
        source_module_op = load_mlir_template(kb, template_file, **kwargs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Generated kernel IR %s:\n%s", function_name, str(source_module_op)
            )
        merger = Merger(
            source_module_op, kb.module_body.owner, target_symbol_table=kb.symbol_table
        )
        merger.merge()
        return kb.symbol_table[function_name]


def load_mlir_template(kb: KernelBuilder, template_file: str, **kwargs) -> Operation:
    """Loads an MLIR template by name.

    The file is loaded relative to the templates/ directory. It is interpolated
    with **kwargs and loaded into the KernelBuilder.
    """
    template_path = TEMPLATES_DIR / template_file
    template_text = template_path.read_text()
    asm = template_text.format(**kwargs)
    module_op = Operation.parse(asm, source_name=str(template_path), context=kb.context)
    return module_op.operation

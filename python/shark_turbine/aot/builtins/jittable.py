# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tracing builtins."""

from typing import Dict, List, Optional, Sequence, Union

import sys

import torch._dynamo as dynamo

from ..builder import ModuleBuilder

from ..procedural import (
    IrValueTensor,
    CallableIntrinsic,
    ProcedureTrace,
)

from ...dynamo.importer import FxImporter
from ...dynamo.passes import (
    turbine_cpu_pass_pipeline,
)

from iree.compiler.ir import (
    FlatSymbolRefAttr,
    FunctionType,
    Operation,
    StringAttr,
    SymbolTable,
    TypeAttr,
)
from iree.compiler.dialects import (
    func as func_d,
)
from iree.compiler.passmanager import (
    PassManager,
)

StringAttrOrStr = Union[StringAttr, str]


class jittable(CallableIntrinsic):
    """Decorator which takes a PyTorch function and makes it callable from tracing.

    It will be internally JIT-ed and exported into the module as needed.
    """

    __slots__ = [
        "wrapped_f",
        "exported_f",
        "function_name",
    ]

    def __init__(
        self,
        wrapped_f,
        *,
        decomposition_table=None,
        constraints=None,
        function_name: Optional[str] = None,
    ):
        self.wrapped_f = wrapped_f
        self.exported_f = dynamo.export(
            wrapped_f,
            aten_graph=True,
            decomposition_table=decomposition_table,
            constraints=constraints,
            assume_static_by_default=True,
            # TODO: Need to do the signature/tree recomposition ourselves.
            same_signature=False,
        )
        self.function_name = function_name if function_name else wrapped_f.__name__

    def __repr__(self):
        return f"<Jittable PyTorch func: {self.exported_f}>"

    def resolve_call(self, proc_trace: ProcedureTrace, *args):
        # TODO: What to do with kwargs?
        # Ask dynamo to give us an aten graph.
        gm, guards = self.exported_f(*args)

        # Import the FX graph to MLIR in a new module.
        fx_importer = FxImporter(context=proc_trace.context)
        fx_importer.import_stateless_graph(gm.graph, func_name=self.function_name)
        print(fx_importer.module, file=sys.stderr)

        # Within the isolated module, convert to MLIR.
        with proc_trace.context:
            pm = PassManager.parse(
                "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline,symbol-dce)"
            )
            pm.run(fx_importer.module.operation)
        print(fx_importer.module.operation, file=sys.stderr)

        # Splice the converted module into the main module by taking advantage
        # of what we know about the conversion module:
        #   1. There will be a public function of `self.function_name` that we
        #      want to preserve a handle to.
        #   2. The function may symbolically refer to other functions and
        #      globals.
        #   3. There is no concept of a module initializer.
        #   4. When allocating tracked globals, we set them up in both the conversion
        #      module and the main module. We note them in the conversion module with
        #      the attribute `util.import_as_symbol` so that we can re-associate here.
        merger = _Merger(
            proc_trace.module_builder,
            fx_importer.module.operation,
            fx_importer.symbol_table,
            self.function_name,
        )
        target_op = merger.merge()
        assert target_op, "Could not find target op in merged module"

        # TODO: Debug upstream why iteration over children isn't creating a typed view.
        # This should just be `target_op.function_type`
        target_ftype = FunctionType(
            TypeAttr(target_op.attributes["function_type"]).value
        )
        target_symbol_ref = FlatSymbolRefAttr.get(
            StringAttr(target_op.attributes["sym_name"]).value
        )

        # TODO: Populate arguments.
        flat_ir_args = []
        assert len(flat_ir_args) == len(target_ftype.inputs), (
            f"Mismatched number of IR call args vs function decl: "
            f"{len(flat_ir_args)} vs {len(target_ftype.inputs)}\n"
            f"  For call to: {target_ftype}"
        )

        with proc_trace.ip, proc_trace.loc:
            flat_ir_results = func_d.CallOp(
                target_ftype.results, target_symbol_ref, flat_ir_args
            ).results

        # TODO: Unflatten the right way with the PyTorch captured schema.
        if len(flat_ir_results) == 0:
            return None
        elif len(flat_ir_results) == 1:
            return IrValueTensor(flat_ir_results[0])
        else:
            return list(map(lambda ir_value: IrValueTensor(ir_value), flat_ir_results))


class _Merger:
    __slots__ = [
        "context",
        "to_module_builder",
        "from_module_op",
        "from_symbol_table",
        "import_function_name",
        "rename_map",
        "nested_symbol_ops",
        "nested_symbol_table_ops",
        "private_attr",
    ]

    def __init__(
        self,
        to_module_builder: ModuleBuilder,
        from_module_op: Operation,
        from_symbol_table: SymbolTable,
        import_function_name: str,
    ):
        self.context = from_module_op.context
        self.to_module_builder = to_module_builder
        self.from_module_op = from_module_op
        self.from_symbol_table = from_symbol_table
        self.import_function_name = import_function_name

        self.rename_map: Dict[StringAttr, StringAttr] = {}
        self.nested_symbol_ops: List[Operation] = []
        self.nested_symbol_table_ops: List[Operation] = []
        self.private_attr = StringAttr.get("private", self.context)

    def merge(self) -> Optional[Operation]:
        # The needle we are looking for.
        imported_func_op: Optional[Operation] = None

        # Import functions.
        func_ops = _get_top_level_ops(self.from_module_op, func_d.FuncOp.OPERATION_NAME)
        for func_op in func_ops:
            # Pre-rename, check if it is the one we are looking for.
            func_name = _get_symbol_name(func_op)
            if func_name == self.import_function_name:
                imported_func_op = func_op
            # All functions become private.
            func_op.attributes["sym_visibility"] = self.private_attr
            self.import_symbol_op(func_op)
            self.nested_symbol_table_ops.append(func_op)

        # Go back through to nested symbol table ops and RAUW.
        for sym_operation in self.nested_symbol_table_ops:
            for from_symbol, to_symbol in self.rename_map.items():
                from_name = StringAttr(from_symbol).value
                to_name = StringAttr(to_symbol).value
                SymbolTable.replace_all_symbol_uses(from_name, to_name, sym_operation)

        return imported_func_op

    def import_symbol_op(self, symbol_op):
        target_symbol_table = self.to_module_builder.symbol_table
        symbol_op = symbol_op.detach_from_parent()
        orig_symbol = SymbolTable.get_symbol_name(symbol_op)
        orig_symbol_name = StringAttr(orig_symbol).value
        # Make sure it is unique.
        new_symbol_name = _uniqueify_name(orig_symbol_name, target_symbol_table)
        if new_symbol_name != orig_symbol_name:
            SymbolTable.set_symbol_name(symbol_op, new_symbol_name)
            self._rename(orig_symbol, new_symbol_name)

        self.to_module_builder.body.append(symbol_op)
        self.nested_symbol_ops.append(symbol_op)
        target_symbol_table.insert(symbol_op)

    def _rename(self, from_symbol: StringAttrOrStr, to_symbol: StringAttrOrStr):
        from_symbol = self._make_string_attr(from_symbol)
        to_symbol = self._make_string_attr(to_symbol)
        if from_symbol != to_symbol:
            self.rename_map[from_symbol] = to_symbol

    def _make_string_attr(self, string_attr_or_str: StringAttrOrStr):
        if isinstance(string_attr_or_str, str):
            with self.context:
                return StringAttr.get(string_attr_or_str)
        else:
            return StringAttr(string_attr_or_str)


def _get_top_level_ops(module_op: Operation, *op_names: str) -> Sequence[Operation]:
    results = []
    for op_view in module_op.regions[0].blocks[0]:
        op = op_view.operation
        if op.name in op_names:
            results.append(op)
    return results


def _get_symbol_name(op: Operation) -> str:
    return StringAttr(op.attributes["sym_name"]).value


def _uniqueify_name(local_name: str, st: SymbolTable) -> str:
    index = -1
    while True:
        index += 1
        full_name = local_name
        if index > 0:
            full_name += f"${index}"
        if full_name not in st:
            return full_name

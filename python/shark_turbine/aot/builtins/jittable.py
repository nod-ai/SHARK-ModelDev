# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tracing builtins."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import sys

import torch
from torch._decomp import get_decompositions
import torch._dynamo as dynamo
from torch.fx import (
    Graph,
)
from torch.fx.passes.shape_prop import TensorMetadata

from ...dynamo.importer import (
    GraphNodeImporter,
    FxImporter,
)

from ...dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
    CPU_DECOMPOSITIONS,
)

from ..support.utils import (
    logger,
    tree_flatten,
    tree_unflatten,
)

from ..support.ir_utils import (
    ModuleBuilder,
)

from ..support.procedural import (
    CallableIntrinsic,
    IrImmediateTensor,
    IrTensor,
    IrTrace,
    MaterializedGlobal,
)

from ..support.ir_imports import (
    FlatSymbolRefAttr,
    FunctionType,
    MLIRError,
    Operation,
    PassManager,
    StringAttr,
    SymbolTable,
    TypeAttr,
    Value,
    func_d,
    util_d,
)

StringAttrOrStr = Union[StringAttr, str]


def _make_literal_resolver(module_builder: ModuleBuilder):
    # When we first encounter a global during import, we have to pull it
    # into the local module being populated by the GraphNodeImporter. This
    # will exactly match the global in the target module we are merging into
    # and exists so that the IR is valid during Fx import. We keep the set of
    # symbols we have done this to here.
    cloned_global_symbols: Set[str] = set()

    def resolver(py_value: Any, gni: GraphNodeImporter) -> Optional[Value]:
        # We support resolution of tracked reference types. Currently this
        # only includes Tensors. All others we let the importer do what it
        # is going to do.
        if not isinstance(py_value, torch.Tensor):
            return None

        # See if we know about it.
        mapping = module_builder.global_ref_tracker.track(py_value)
        if mapping.is_empty:
            # If it is unknown, just let the default importer take it on.
            return None

        # Already materialized.
        logger.debug("Resolved defined global for literal %r", mapping)
        materialized_global: MaterializedGlobal = mapping.value  # type: ignore

        # Clone the global into the import module (so that our symbol refs are
        # legal). Note that the merger will ignore these since they already
        # exist in the target module.
        if materialized_global.symbol_name not in cloned_global_symbols:
            materialized_global.global_op.operation.clone(ip=gni.fx_importer._m_ip)
            cloned_global_symbols.add(materialized_global.symbol_name)

        # Emit a global load and conversion.
        vtensor_type = gni._cc.tensor_to_vtensor_type(py_value)
        loaded_value = util_d.GlobalLoadOp(
            materialized_global.ir_type, materialized_global.symbol_name
        ).result
        converted_value = Operation.create(
            "torch_c.from_builtin_tensor",
            results=[vtensor_type],
            operands=[loaded_value],
        ).result
        return converted_value

    return resolver


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
        decompose_ops: Optional[List[torch._ops.OpOverload]] = None,
        decomposition_table: Optional[
            Dict[torch._ops.OpOverload, Callable[..., Any]]
        ] = None,
        constraints=None,
        function_name: Optional[str] = None,
    ):
        if decomposition_table is None:
            decomposition_table = {}
        if decompose_ops is None:
            decompose_ops = DEFAULT_DECOMPOSITIONS + CPU_DECOMPOSITIONS

        if decompose_ops:
            decomposition_table.update(get_decompositions(decompose_ops))

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

    def resolve_call(self, proc_trace: IrTrace, *py_args, **py_kwargs):
        flat_py_args, args_tree = tree_flatten((py_args, py_kwargs))

        # Convert procedural trace values to things that Dynamo can handle.
        flat_pytorch_args = []
        flat_ir_args = []
        for py_arg in flat_py_args:
            ir_arg, pytorch_arg = self._split_py_arg(py_arg)
            flat_ir_args.append(ir_arg)
            flat_pytorch_args.append(pytorch_arg)

        # Ask dynamo to give us an aten graph.
        pytorch_args, pytorch_kwargs = tree_unflatten(flat_pytorch_args, args_tree)
        gm, guards = self.exported_f(*pytorch_args, **pytorch_kwargs)
        # We capture metadata about the results from the raw graph so that we can
        # pass it along in the trace (since the IREE type system is a partial erasure
        # of the PyTorch type system and we need the fidelity).
        # This could be done by the importer but the API gets twisty so just
        # doing it here since it isn't clear anyone else would ever want this.
        pytorch_meta_results = _extract_graph_output_metadata(gm.graph)

        # Import the FX graph to MLIR in a new module.
        fx_importer = FxImporter(
            context=proc_trace.context,
            config_check=False,
            literal_resolver_callback=_make_literal_resolver(proc_trace.module_builder),
        )
        fx_importer.import_stateless_graph(gm.graph, func_name=self.function_name)

        # TODO: Real debugging options
        # print(fx_importer.module, file=sys.stderr)

        # Within the isolated module, convert to MLIR.
        with proc_trace.context as context:
            try:
                pm = PassManager.parse("builtin.module(torch-to-iree)")
                # Uncomment these two lines to debug.
                # TODO: Real debug options.
                # context.enable_multithreading(False)
                # pm.enable_ir_printing()
                # Run.
                pm.run(fx_importer.module.operation)
            except MLIRError:
                # TODO: Better error handling.
                print(fx_importer.module.operation, file=sys.stderr)
                raise

        # Uncomment to print the final module.
        # TODO: Real debugging options.
        # print(fx_importer.module.operation, file=sys.stderr)

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

        assert len(flat_ir_args) == len(target_ftype.inputs), (
            f"Mismatched number of IR call args vs function decl: "
            f"{len(flat_ir_args)} vs {len(target_ftype.inputs)}\n"
            f"  For call to: {target_ftype}"
        )

        with proc_trace.ip, proc_trace.loc:
            flat_ir_results = func_d.CallOp(
                target_ftype.results, target_symbol_ref, flat_ir_args
            ).results

        flat_py_results = []
        for ir_result, pytorch_meta in zip(flat_ir_results, pytorch_meta_results):
            if isinstance(pytorch_meta, TensorMetadata):
                flat_py_results.append(IrImmediateTensor(ir_result, pytorch_meta.dtype))
            else:
                raise TypeError(
                    f"Unknown PyTorch->IREE value mapping for jittable result: {pytorch_meta}->{ir_result}"
                )

        # TODO: Unflatten the right way with the PyTorch captured schema.
        if len(flat_py_results) == 0:
            return None
        elif len(flat_py_results) == 1:
            return flat_py_results[0]
        else:
            return flat_py_results

    def _split_py_arg(self, arg) -> Tuple[Value, Any]:
        if isinstance(arg, IrTensor):
            return arg.ir_value, arg._to_meta_tensor()

        raise TypeError(f"Unsupported argument to jittable: {arg}")


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


def _extract_graph_output_metadata(g: Graph) -> List[Union[None, TensorMetadata]]:
    output_metadata = []
    for node in g.nodes:
        if node.op == "output":
            # An output node's args[0] is the return value. This seems to
            # always be "boxed" as a tuple, which we emit as multi-results.
            for result_node in node.args[0]:
                if result_node is None:
                    output_metadata.append(None)
                else:
                    tensor_meta = result_node.meta.get("tensor_meta")
                    assert tensor_meta is None or isinstance(
                        tensor_meta, TensorMetadata
                    )
                    output_metadata.append(tensor_meta)
    return output_metadata

# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import operator
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from iree.compiler.ir import (
    Attribute as MlirAttribute,
    Block,
    Context,
    FloatAttr,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    Operation,
    StringAttr,
    Type as MlirType,
    Value,
)

import iree.compiler.dialects.func as func_dialect

# import iree.compiler.dialects.torch as torch_dialect


import torch
import torch.fx as torch_fx
from torch.fx.passes.shape_prop import TensorMetadata

from torch import (
    dtype as TorchDtype,
    FunctionSchema,
)

from torch._ops import (
    OpOverload as TorchOpOverload,
)

from torch._subclasses import (
    FakeTensor as TorchFakeTensor,
)

from torch.fx import (
    Graph,
    GraphModule,
)

from torch.fx.node import (
    Argument as NodeArgument,
)

__all__ = [
    "FxImporter",
]

REQUIRED_DIALCTS = [
    "builtin",
    "func",
    "torch",
]

TORCH_DTYPE_TO_MLIR_TYPE_ASM = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.uint8: "ui8",
    torch.int8: "si8",
    torch.int16: "si16",
    torch.int32: "si32",
    torch.int64: "si64",
    torch.bool: "i1",
    torch.qint8: "!torch.qint8",
    torch.quint8: "!torch.quint8",
    torch.complex32: "complex<f16>",
    torch.complex64: "complex<f32>",
    torch.complex128: "complex<f64>",
}


class FxImporter:
    """Main entry-point for importing an fx.GraphModule."""

    __slots__ = [
        "_c",
        "_cc",
        "_m",
        "_m_ip",
    ]

    def __init__(
        self,
        module: Optional[Module] = None,
        context: Optional[Context] = None,
        config_check: bool = True,
    ):
        if module is not None:
            assert context is None, "If configuring with a Module, context must be None"
            self._m = module
            self._c = self.module.context
        else:
            self._c = context if context else Context()
            self._m = Module.create(Location.unknown(self._c))
        if config_check:
            # Production code can disable this for a bit of a boost.
            self._config_check()
        self._cc = ContextCache(self._c)
        self._m_ip = InsertionPoint(self._m.body)

    def _config_check(self):
        for dname in REQUIRED_DIALCTS:
            try:
                self._c.dialects[dname]
                logging.debug("Context has registered dialect '%s'", dname)
            except IndexError:
                raise RuntimeError(
                    f"The MLIR context {self._c} is missing required dialect '{dname}'"
                )

    @property
    def module(self) -> Module:
        return self._m

    def import_graph_module(self, gm: GraphModule):
        self.import_stateless_graph(gm.graph)

    def import_stateless_graph(self, g: Graph, func_name: str = "main"):
        ftype, loc = self._graph_to_function_meta(g)
        # TODO: The FuncOp constructor requires a context-manager context.
        # Fix upstream and then unnest.
        with loc:
            func = func_dialect.FuncOp(
                func_name,
                ftype,
                ip=self._m_ip,
            )
            entry_block = Block.create_at_start(func.body, ftype.inputs)
        node_importer = GraphNodeImporter(self._c, self._cc, entry_block)
        node_importer.import_nodes(g.nodes)

    def _graph_to_function_meta(self, g: Graph) -> Tuple[FunctionType, Location]:
        """Extracts function metadata from the Graph.

        Principally, this includes the FunctionType, but in the future,
        it should also return other annotations (input strides, etc) that
        affect compilation and should be included as arg attrs.
        """
        input_types = []
        result_types = []
        loc = None
        for node in g.nodes:
            # Assume that the first node we can get a location for is about as
            # good as it gets as an overall function location.
            if loc is None:
                loc = self._cc.get_node_location(node)
            if node.op == "placeholder":
                input_types.append(self._cc.node_val_to_type(node))
            elif node.op == "output":
                # An output node's args[0] is the return value. This seems to
                # always be "boxed" as a tuple, which we emit as multi-results.
                for result_node in node.args[0]:
                    result_types.append(self._cc.node_val_to_type(result_node))
        return (
            FunctionType.get(input_types, result_types, context=self._c),
            loc if loc else Location.unknown(self._c),
        )


class ContextCache:
    """Caches per-context lookups of various things that we ask for repeatedly."""

    __slots__ = [
        "_c",
        "_dtype_to_type",
        "_tensor_metadata_cache",
        # Types.
        "torch_bool_type",
        "torch_float_type",
        "torch_int_type",
        "torch_none_type",
        "torch_str_type",
    ]

    def __init__(self, context: Context):
        self._c = context
        self._dtype_to_type: Dict[TorchDtype, MlirType] = {}
        self._tensor_metadata_cache: Dict[Tuple[torch.Size, torch.dtype], MlirType] = {}

        # Common types.
        with context:
            self.torch_bool_type = MlirType.parse("!torch.bool")
            self.torch_float_type = MlirType.parse("!torch.float")
            self.torch_int_type = MlirType.parse("!torch.int")
            self.torch_none_type = MlirType.parse("!torch.none")
            self.torch_str_type = MlirType.parse("!torch.str")

    def integer_attr(self, value: int, bits: int) -> MlirAttribute:
        c = self._c
        return IntegerAttr.get(IntegerType.get_signless(bits, c), value)

    def node_val_to_type(self, node: torch_fx.Node) -> MlirType:
        try:
            tensor_meta = node.meta.get("tensor_meta")
            if tensor_meta is not None:
                assert isinstance(tensor_meta, TensorMetadata)
                # TODO: We should probably only be doing this if "vanilla".
                # Specifically, there are strides/qparams/etc on there that
                # should be annotated somewhere.
                return self.tensor_metadata_to_type(tensor_meta)
            else:
                raise NotImplementedError(
                    f"FIXME: Unsupported placeholder node (this often indicates that a necessary) "
                    f"fx preprocessing pass was not run): {node.meta}"
                )
        except KeyError as e:
            raise RuntimeError(
                f"FIXME: Illegal access to torch.fx.Node.meta: {e} ({node.meta.keys()} : {node.meta})"
            )

    def tensor_metadata_to_type(self, tm: TensorMetadata) -> MlirType:
        key = (tm.shape, tm.dtype)
        t = self._tensor_metadata_cache.get(key)
        if t is None:
            shape_asm = ",".join(str(d) for d in tm.shape)
            mlir_type = self.dtype_to_type(tm.dtype)
            t = MlirType.parse(
                f"!torch.vtensor<[{shape_asm}],{str(mlir_type)}>", context=self._c
            )
            self._tensor_metadata_cache[key] = t
        return t

    def dtype_to_type(self, dtype: TorchDtype) -> MlirType:
        t = self._dtype_to_type.get(dtype)
        if t is None:
            try:
                asm = TORCH_DTYPE_TO_MLIR_TYPE_ASM[dtype]
            except IndexError:
                raise ValueError(f"Unknown conversion from {dtype} to IREE type")
            t = MlirType.parse(asm, self._c)
            self._dtype_to_type[dtype] = t
        return t

    def get_node_location(self, node: torch_fx.Node) -> Optional[Location]:
        stack_trace = node.meta.get("stack_trace")
        if stack_trace is None:
            return None
        # Ugh.
        # TODO: Avoid needing to regex match this.
        # https://github.com/pytorch/pytorch/issues/91000
        m = re.search(r"""File "([^"]+)", line ([0-9]+),""", node.stack_trace)
        filename, line = m.group(1), int(m.group(2))
        return Location.file(filename, line, col=0, context=self._c)


class GraphNodeImporter:
    """Imports graph nodes into an MLIR function.

    The caller must have already created the function.
    """

    __slots__ = [
        "_b",
        "_c",
        "_cc",
        "_v",
        "_multi_result_nodes",
    ]

    def __init__(self, context: Context, context_cache: ContextCache, block: Block):
        self._c = context
        self._cc = context_cache
        self._b = block
        # Map of (Node, result_index) to MLIR Value.
        self._v: Dict[Tuple[torch_fx.Node, int], Value] = {}
        # Statically multi-result nodes which we have de-tupled are noted here.
        # They will have their getitem calls short-circuited.
        self._multi_result_nodes: Set[torch_fx.Node] = set()

    def import_nodes(self, nodes: Sequence[torch_fx.Node]):
        with InsertionPoint(self._b):
            loc = Location.unknown()
            num_placeholders = 0
            for node in nodes:
                op = node.op
                # Attempt to extract locations. Not everything has them,
                # so we do our best.
                new_loc = self._cc.get_node_location(node)
                if new_loc is not None:
                    loc = new_loc
                if op == "placeholder":
                    # Associate the placeholder node with corresponding block
                    # argument.
                    self._v[(node, 0)] = self._b.arguments[num_placeholders]
                    num_placeholders += 1
                elif op == "call_function":
                    target = node.target
                    if target == operator.getitem:
                        # Special case handling of getitem for when it is resolving
                        # against a function call that we know has returned multiple
                        # results. We short-circuit this case because we have modeled
                        # function calls to natively return multiple results vs tupling.
                        getitem_ref, getitem_index = node.args
                        if getitem_ref in self._multi_result_nodes:
                            try:
                                self._v[(node, 0)] = self._v[
                                    (getitem_ref, getitem_index)
                                ]
                            except IndexError:
                                raise RuntimeError(
                                    f"getitem de-aliasing failed. This likely "
                                    f"indicates a programmer error that usually "
                                    f"would have happened at runtime. Please "
                                    f"notify developers if this case happens "
                                    f"(at {loc})."
                                )
                        else:
                            raise NotImplementedError(
                                f"General getitem access to non-multi-result ops"
                            )
                    elif isinstance(target, TorchOpOverload):
                        # Dispatch to an ATen op.
                        self._import_torch_op_overload(loc, node, target)
                    else:
                        raise NotImplementedError(
                            f"FIX ME: Unimplemented call_function: target={node.target}, {node.meta}"
                        )
                elif op == "output":
                    # args[0] is a singleton tuple that we flatten into multiple
                    # results.
                    operands = [self._import_argument(arg) for arg in node.args[0]]
                    func_dialect.ReturnOp(operands, loc=loc)

    def _import_torch_op_overload(
        self, loc: Location, node: torch_fx.Node, target: TorchOpOverload
    ):
        schema = target._schema
        assert isinstance(schema, FunctionSchema)

        # Map to a `torch` dialect name.
        namespace, sep, unqualified_name = schema.name.partition("::")
        assert sep, f"Malformed Torch op name {schema.name}"
        mlir_op_name = f"torch.{namespace}.{unqualified_name}"
        if schema.overload_name != "":
            mlir_op_name += f".{schema.overload_name}"

        if not self._c.is_registered_operation(mlir_op_name):
            # TODO: Implement a config setting to allow these to flow through.
            raise NotImplementedError(
                f"Unimplemented torch op in the IREE compiler: '{mlir_op_name}' "
                f"(either implement this op/variant or configure the compiler to "
                f"allow unknown operations and fallback to PyTorch)."
            )

        return_count = len(schema.returns)
        if return_count == 1:
            # Unary return directly maps a single meta["val"] and cannot be subscripted.
            result_types = [self._cc.tensor_metadata_to_type(node.meta["tensor_meta"])]
        elif return_count == 0:
            # TODO: Implement.
            raise NotImplementedError("FIXME: Zero ATen results")
        else:
            # Multi-return will unpack the meta["val"] and trigger our getitem subscripting
            # short-circuit above. Note that if we ever choose to also fully reify Python
            # level result tuples, we will need to create a tuple-boxed version of this and
            # redirect to it for generic object access.
            self._multi_result_nodes.add(node)
            raise NotImplementedError("FIXME: Multiple ATen results")

        # Unroll operands from formal parameters, args and kwargs.
        operands = []
        for i, parameter in enumerate(schema.arguments):
            if parameter.kwarg_only and parameter.name in node.kwargs:
                # TODO: Nice error if KeyError.
                operands.append(self._import_argument(node.kwargs[parameter.name], loc))
            elif i < len(node.args):
                arg = node.args[i]
                operands.append(self._import_argument(node.args[i], loc))
            else:
                operands.append(
                    self._import_default_value(
                        loc, parameter.default_value, parameter.type
                    )
                )

        operation = Operation.create(
            mlir_op_name,
            results=result_types,
            operands=operands,
            loc=loc,
        )

        # Record value mapping.
        for i, value in enumerate(operation.results):
            self._v[(node, i)] = value

    def _import_argument(self, arg: NodeArgument, loc: Location = None) -> Value:
        """Import an FX `Argument`, which must result to an MLIR `Value`."""
        if isinstance(arg, torch_fx.Node):
            # If implementing boxed support for multi-result nodes, then
            # this will need to do something more intelligent.
            if arg in self._multi_result_nodes:
                raise RuntimeError(f"Attempt to de-reference a multi-result node")
            return self._v[(arg, 0)]
        elif isinstance(arg, torch_fx.immutable_collections.immutable_list):
            # create list operands
            list_operands = []
            for operand in arg:
                if not isinstance(operand, type(arg[0])):
                    raise TypeError(f"Lists with multiple types are not supported, got: {type(input[0])}, {type(operand)}")

                val = self._import_default_value(loc, operand, SCALAR_TYPE_TO_TORCH_TYPE[type(operand)])
                list_operands.append(val)

            # construct list op
            result_type = MlirType.parse(SCALAR_TYPE_TO_TORCH_LIST_TYPE[type(arg[0])], context=self._c)
            operation = Operation.create(
                "torch.prim.ListConstruct",
                results=[result_type],
                operands=list_operands,
                loc=loc,
            )
            return operation.result
        elif type(arg) in LITERAL_CONVERTER_MAP._cache:
            with loc:
                arg_value = LITERAL_CONVERTER_MAP.lookup(type(arg))(arg, self, self._cc)
            return arg_value
        else:
            raise NotImplementedError(f"FIXME: Unsupported Node Argument: {arg}")

    def _import_default_value(self, loc: Location, arg, expected_jit_type) -> Value:
        """Imports a defaulted value for a known function schema."""
        cvt = LITERAL_CONVERTER_MAP.lookup(type(arg))
        if cvt is None:
            raise RuntimeError(f"Unhandled default value ({arg.__class__}): {arg})")
        with loc:
            return cvt(arg, self, self._cc)

        # TODO: Support torch specific types which show up in function schemas.
        # These all require an expected_jit_type to convert.
        # torch.dtype, torch.device, torch.memory_format, torch.layout
        # list


class TypeSubclassMap:
    """Mapping of super-types to values.

    Maintains a cache of actual types seen and uses that instead of a linear
    scan.
    """

    __slots__ = [
        "_cache",
        "_mapping",
    ]

    def __init__(self):
        # The linear list of converters.
        self._mapping: List[Tuple[type, Any]] = []
        # When there is a hit on the linear mapping, memoize it here.
        self._cache: Dict[type, Any] = {}

    def map(self, t: type, value: Any):
        self._mapping.append((t, value))
        self._cache[t] = value

    def lookup(self, t: type) -> Any:
        try:
            return self._cache[t]
        except KeyError:
            pass
        for t_super, value in self._mapping:
            if issubclass(t, t_super):
                self._cache[t] = value
                return value
        else:
            self._cache[t] = None
            return None


def _make_constant_op(
    op_name: str, value_attr: MlirAttribute, result_type: Optional[MlirType] = None
) -> Operation:
    return Operation.create(
        op_name,
        results=[result_type if result_type else value_attr.type],
        attributes={"value": value_attr},
    )


LITERAL_CONVERTER_MAP = TypeSubclassMap()
LITERAL_CONVERTER_MAP.map(
    type(None),
    lambda arg, gni, cc: Operation.create(
        "torch.constant.none", results=[cc.torch_none_type]
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    bool,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.bool", cc.integer_attr(arg, 1), cc.torch_bool_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    int,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.int", cc.integer_attr(arg, 64), cc.torch_int_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    float,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.float", FloatAttr.get_f64(arg), cc.torch_float_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    str,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.str", StringAttr.get(arg), cc.torch_str_type
    ).result,
)

SCALAR_TYPE_TO_TORCH_LIST_TYPE = {
    int: "!torch.list<int>",
    float: "!torch.list<float>",
    str: "!torch.list<str>",
}

SCALAR_TYPE_TO_TORCH_TYPE = {
    int: "!torch.int",
    float: "!torch.float",
    str: "!torch.str"
}
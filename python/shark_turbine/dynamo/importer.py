# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import builtins
import logging
import operator
import re
from types import NoneType
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

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
    DenseElementsAttr,
)

import iree.compiler.dialects.func as func_dialect
from iree.compiler.ir import SymbolTable

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

TORCH_DTYPE_TO_NPY_TYPE = {
    # torch.qint8: None, # no equivalent np datatype
    # torch.quint8: None,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    # torch.bf16: None, there's no equivalent np datatype so this isn't supported right now
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bool: np.bool_,
    # torch.complex32: None, # no equivalent precision for numpy
    # torch.complex64: np.complex64, # complex dtypes can't be parsed by DenseElementsAttr in the numpy buffer format
    # torch.complex128: np.complex128,
}

# https://github.com/llvm/torch-mlir/blob/4c24472dea1c9102b898768b0b11e31487e50207/python/torch_mlir/_dynamo_fx_importer.py#L189
TORCH_DTYPE_TO_INT = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    # torch.complex_half 8
    torch.complex32: 9,
    torch.complex64: 10,
    torch.bool: 11,
    # torch.qint8: 12, # quantized dtypes are not supported in all backends, currently we do not support them
    # torch.quint8: 13,
    # torch.qint32 14
    torch.bfloat16: 15,
}

# https://github.com/llvm/torch-mlir/blob/4c24472dea1c9102b898768b0b11e31487e50207/python/torch_mlir/_dynamo_fx_importer.py#L223
TORCH_MEMORY_FORMAT_TO_INT = {
    torch.contiguous_format: 0,
    torch.preserve_format: 1,
    torch.channels_last: 2,
    torch.channels_last_3d: 3,
}

# https://github.com/llvm/torch-mlir/blob/4c24472dea1c9102b898768b0b11e31487e50207/python/torch_mlir/_dynamo_fx_importer.py#L235
TORCH_LAYOUT_TO_INT = {
    torch.strided: 0,
    torch.sparse_coo: 1,
    torch.sparse_csr: 2,
    torch.sparse_csc: 3,
    torch.sparse_bsr: 4,
    torch.sparse_bsc: 5,
}


class FxImporter:
    """Main entry-point for importing an fx.GraphModule."""

    __slots__ = [
        "_c",
        "_cc",
        "_m",
        "_m_ip",
        "symbol_table",
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
        self.symbol_table = SymbolTable(self._m.operation)

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
        self.symbol_table.insert(func)

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
                    if result_node is None:
                        result_types.append(
                            MlirType.parse("!torch.none", context=self._c)
                        )
                    else:
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
        "torch_device_type",
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
            self.torch_device_type = MlirType.parse("!torch.Device")

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
                    operands = [self._import_argument(loc, arg) for arg in node.args[0]]
                    func_dialect.ReturnOp(operands, loc=loc)

    def _import_torch_op_overload(
        self, loc: Location, node: torch_fx.Node, target: TorchOpOverload
    ):
        # replace lift_fresh_copy with clone op
        if target == torch.ops.aten.lift_fresh_copy.default:
            node.target = target = torch.ops.aten.clone.default
            node.args = (node.args[0], None)

        schema = target._schema
        assert isinstance(schema, FunctionSchema)

        # Map to a `torch` dialect name.
        namespace, sep, unqualified_name = schema.name.partition("::")
        assert sep, f"Malformed Torch op name {schema.name}"
        mlir_op_name = f"torch.{namespace}.{unqualified_name}"
        if schema.overload_name != "":
            mlir_op_name += f".{schema.overload_name}"

        # Intervening to use Scalar ops due to incorrect ops from AOT-autograd with scalar arguments.
        if mlir_op_name in TENSOR_SCALAR_OP_CONVERTER and (
            isinstance(node.args[1], float) or isinstance(node.args[1], int)
        ):
            mlir_op_name = TENSOR_SCALAR_OP_CONVERTER[mlir_op_name]

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
            # if "tensor_meta" is None, this will throw unsupported placeholder node error
            result_types = [self._cc.node_val_to_type(node)]
        elif return_count == 0:
            # TODO: Implement.
            raise NotImplementedError("FIXME: Zero ATen results")
        else:
            # Multi-return will unpack the meta["val"] and trigger our getitem subscripting
            # short-circuit above. Note that if we ever choose to also fully reify Python
            # level result tuples, we will need to create a tuple-boxed version of this and
            # redirect to it for generic object access.

            result_types = []
            for v in node.meta["val"]:
                result_types.append(self._cc.tensor_metadata_to_type(v))
            result_types = tuple(result_types)

            self._multi_result_nodes.add(node)
        # Unroll operands from formal parameters, args and kwargs.
        operands = []
        for i, parameter in enumerate(schema.arguments):
            if parameter.kwarg_only and parameter.name in node.kwargs:
                # TODO: Nice error if KeyError.
                operands.append(
                    self._import_argument(
                        loc, node.kwargs[parameter.name], parameter.type
                    )
                )
            elif i < len(node.args):
                operands.append(
                    self._import_argument(loc, node.args[i], parameter.type)
                )
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

    def _import_argument(
        self, loc: Location, arg: NodeArgument, expected_jit_type=None
    ) -> Value:
        """Import an FX `Argument`, which must result to an MLIR `Value`."""
        if isinstance(arg, torch_fx.Node):
            # If implementing boxed support for multi-result nodes, then
            # this will need to do something more intelligent.
            if arg in self._multi_result_nodes:
                raise RuntimeError(f"Attempt to de-reference a multi-result node")

            # catch references to dynamically created constant attributes and make sure they have an origin in our module
            if arg.op == "get_attr" and (arg.target, 0) not in self._v:
                gm = arg.graph.owning_module
                assert hasattr(
                    gm, arg.target
                ), f"Attempting to retrieve attribute '{arg.target}' from module, but no such attribute exists"
                obj = getattr(gm, arg.target)
                with loc:
                    value = LITERAL_CONVERTER_MAP.lookup(type(obj))(obj, self, self._cc)
                self._v[(arg, 0)] = value

            return self._v[(arg, 0)]
        elif isinstance(arg, torch_fx.immutable_collections.immutable_list):
            return self._import_list_argument(loc, arg, expected_jit_type)
        elif type(arg) in LITERAL_CONVERTER_MAP._cache:
            with loc:
                arg_value = LITERAL_CONVERTER_MAP.lookup(type(arg))(arg, self, self._cc)
            return arg_value
        else:
            raise NotImplementedError(f"FIXME: Unsupported Node Argument: {arg}")

    def _import_list_argument(
        self, loc: Location, arg: NodeArgument, expected_jit_type
    ) -> Value:
        assert (
            isinstance(expected_jit_type, torch.ListType)
            or (
                isinstance(expected_jit_type, torch.OptionalType)
                and isinstance(expected_jit_type.getElementType(), torch.ListType)
            )
            or isinstance(expected_jit_type, NoneType)
        ), f"Unexpected jit type as list argument: {arg} of type {expected_jit_type}"

        # parse list type
        if expected_jit_type is None:
            element_type = type(arg[0])
        else:
            element_jit_type = expected_jit_type.getElementType()

            # this branch is needed to handle Optional[List[]] types
            if isinstance(element_jit_type, torch.ListType):
                element_jit_type = element_jit_type.getElementType()

            # this handles getting the inner types for List[Optional[]] types
            is_optional_type = isinstance(element_jit_type, torch.OptionalType)
            if is_optional_type:
                element_jit_type = element_jit_type.getElementType()
            element_type = TORCH_TYPE_TO_PY_TYPE[type(element_jit_type)]

        # create list operands
        list_operands = []

        for operand in arg:
            operand_type = type(operand)
            if isinstance(operand, torch.fx.Node):
                if operand in self._multi_result_nodes:
                    raise RuntimeError(f"Attempt to de-reference a multi-result node")
                val = self._v[(operand, 0)]
                val_type = str(val.type)
                assert (
                    isinstance(element_type, str) and element_type in val_type
                ), f"Heterogeneous lists are not supported: expected {element_type}, got {val_type}"
            else:
                assert (is_optional_type and operand_type is NoneType) or (
                    element_type == operand_type
                ), f"Heterogeneous lists are not supported: expected {element_type}, got {operand_type}"

                operand_jit_type = (
                    torch.NoneType if operand_type is NoneType else element_jit_type
                )
                val = self._import_default_value(loc, operand, operand_jit_type)

            list_operands.append(val)

        # construct list op
        if is_optional_type:
            list_type = PY_TYPE_TO_TORCH_OPTIONAL_LIST_TYPE[element_type]
        else:
            list_type = PY_TYPE_TO_TORCH_LIST_TYPE[element_type]

        result_type = MlirType.parse(list_type, context=self._c)
        operation = Operation.create(
            "torch.prim.ListConstruct",
            results=[result_type],
            operands=list_operands,
            loc=loc,
        )

        return operation.result

    def _import_default_value(self, loc: Location, arg, expected_jit_type) -> Value:
        """Imports a defaulted value for a known function schema."""
        if isinstance(arg, list):
            return self._import_list_argument(loc, arg, expected_jit_type)

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


def _make_vtensor_literal_op(tensor: torch.Tensor, mlir_type: MlirType) -> Operation:
    npy_dtype = TORCH_DTYPE_TO_NPY_TYPE.get(tensor.dtype)
    assert (
        npy_dtype is not None
    ), f"Can not create literal tensor for unsupported datatype: {tensor.dtype}"
    # We need a raw buffer of data in order to create an ElementsAttr for the invocation of torch.vtensor.literal,
    # but torch.Tensor does not fulfill the python buffer/array interface hence we must convert to a numpy array to get
    # a raw buffer of our data. We can't call torch.Tensor.numpy() directly because this internally forces a call to
    # detach() which throws an error as we are operating in a FakeTensorMode, hence the simplest way to get this raw
    # buffer is via the indirection: Tensor -> list -> numpy array. This allows us to create a vtensor literal as
    # desired, but also limits which data types we can support in this function (see TORCH_DTYPE_TO_NPY_TYPE above)
    np_tensor = np.array(tensor.tolist()).astype(npy_dtype)
    bytes = memoryview(np_tensor)

    mlir_asm_type = str(mlir_type)
    tensor_type = MlirType.parse(
        f"!torch.vtensor<{list(tensor.size())},{mlir_asm_type}>"
    )
    elements_attr = DenseElementsAttr.get(bytes, signless=False)
    return Operation.create(
        name="torch.vtensor.literal",
        results=[tensor_type],
        attributes={"value": elements_attr},
    )


LITERAL_CONVERTER_MAP = TypeSubclassMap()
LITERAL_CONVERTER_MAP.map(
    NoneType,
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
LITERAL_CONVERTER_MAP.map(
    torch.Tensor,
    lambda arg, gni, cc: _make_vtensor_literal_op(
        arg, cc.dtype_to_type(arg.dtype)
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    torch.device,
    lambda arg, gni, cc: _make_constant_op(
        "torch.constant.device", StringAttr.get(str(arg)), cc.torch_device_type
    ).result,
)
LITERAL_CONVERTER_MAP.map(
    torch.dtype,
    lambda arg, gni, cc: LITERAL_CONVERTER_MAP.lookup(int)(
        TORCH_DTYPE_TO_INT[arg], gni, cc
    ),
)
LITERAL_CONVERTER_MAP.map(
    torch.layout,
    lambda arg, gni, cc: LITERAL_CONVERTER_MAP.lookup(int)(
        TORCH_LAYOUT_TO_INT[arg], gni, cc
    ),
)
LITERAL_CONVERTER_MAP.map(
    torch.memory_format,
    lambda arg, gni, cc: LITERAL_CONVERTER_MAP.lookup(int)(
        TORCH_MEMORY_FORMAT_TO_INT[arg], gni, cc
    ),
)

TORCH_TYPE_TO_PY_TYPE = {
    torch.IntType: int,
    torch.FloatType: float,
    torch.StringType: str,
    torch.BoolType: bool,
    torch.TensorType: "vtensor",
}

PY_TYPE_TO_TORCH_LIST_TYPE = {
    int: "!torch.list<int>",
    float: "!torch.list<float>",
    str: "!torch.list<str>",
    bool: "!torch.list<bool>",
    "tensor": "!torch.list<tensor>",
    "vtensor": "!torch.list<vtensor>",
}

PY_TYPE_TO_TORCH_OPTIONAL_LIST_TYPE = {
    int: "!torch.list<optional<int>>",
    float: "!torch.list<optional<float>>",
    str: "!torch.list<optional<str>>",
    bool: "!torch.list<optional<bool>>",
    "tensor": "!torch.list<optional<tensor>>",
    "vtensor": "!torch.list<optional<vtensor>>",
}

SCALAR_TYPE_TO_TORCH_TYPE = {
    int: "!torch.int",
    float: "!torch.float",
    str: "!torch.str",
    bool: "!torch.bool",
    NoneType: "!torch.none",
}

# AOT-autograd sometimes falsely emit tensor version op with scalar arguments.
# We may remove this dictionary, if we fix such behavior in the backend.
TENSOR_SCALAR_OP_CONVERTER = {
    "torch.aten.mul.Tensor": "torch.aten.mul.Scalar",
    "torch.aten.div.Tensor": "torch.aten.div.Scalar",
    "torch.aten.add.Tensor": "torch.aten.add.Scalar",
    "torch.aten.sub.Tensor": "torch.aten.sub.Scalar",
    "torch.aten.floor_divide": "torch.aten.floor_divide.Scalar",
}

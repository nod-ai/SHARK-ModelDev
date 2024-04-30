from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Type, TypedDict, Union
import torch.fx as fx
from .regions import RegionGraph
from ..lang.functional_types import Memory, Register
from .._support.indexing import IndexExpr
from .._support.dtype import DataType


class NodeType(TypedDict):
    name: str
    type: Type["CustomNode"]


nodeTypes: NodeType = NodeType()


def get_node_name(string: str, skip_first: bool = True):
    snakeString = ""
    if skip_first:
        snakeString += string[0].lower()
        string = string[1:]
    for i in string:
        if i.isupper():
            snakeString += "_" + i.lower()
        else:
            snakeString += i
    # Drop the "_node" suffix
    return snakeString[:-5]


@dataclass
class CustomNode(ABC):
    """
    This is the base class for all custom fx nodes.
    """

    graph: fx.Graph
    op: Any

    @classmethod
    def from_fx_node(cls, node: fx.Node):
        instance = cls(node.graph, node.op, *node.args)
        instance.fx_node = node
        return instance

    def __str__(self) -> str:
        name = get_node_name(self.__class__.__name__)
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][2:]
        vars_str = ", ".join(vars_list)
        return f"{name} {vars_str}" + "\n"

    def custom_string(self, value_map: dict[str, str]) -> str:
        # If a subclass does not define custom printing we revert to the default
        return str(self)

    def emit(self):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = self.graph.create_proxy(
            "call_function",
            target=self.op,
            args=arg_list,
            kwargs={},
        )

    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name
        return self.fx_node.name


@dataclass
class UnknownNode(CustomNode):
    args: Sequence[Any]
    kwargs: dict[Any, Any]

    @classmethod
    def from_fx_node(cls, node: fx.Node):
        kwargs = node.kwargs | node.meta
        return cls(node.graph, node.op, node.args, kwargs)


# Nodes modeling TKL operations in the kernel language


@dataclass
class ConstructRegisterFromMetadataNode(CustomNode):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    value: float

    def custom_string(self, value_map: dict[str, str]) -> str:
        # TODO: don't rely on extracting data from the fx_node. Everything required
        #       should be a field of this class
        simt_shape = None
        if self.fx_node and "simt_shape" in self.fx_node.meta:
            simt_shape = self.fx_node.meta["simt_shape"]
        return f"construct_register_from_metadata [value = {self.value}] -> Register<{self.shape}, {self.dtype}> -> Register<{simt_shape}, {self.dtype}>\n"


# TODO: fx.Proxy should be wrapped to provide a nice interface to our memory objects as well


@dataclass
class MmaNode(CustomNode):
    lhs: fx.Node
    rhs: fx.Node
    acc: fx.Node

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        lhs = getNode(self.lhs)
        rhs = getNode(self.rhs)
        acc = getNode(self.acc)

        return f"{name} %{get_name(lhs, value_map)}, %{get_name(rhs, value_map)}, %{get_name(acc, value_map)} : {reg(lhs)}, {reg(rhs)} -> {reg(acc)}\n"


def get_name(node: CustomNode, value_map: dict[str, str]) -> str:
    if hasattr(node, "name") and node.name in value_map:
        return value_map[node.name]
    if hasattr(node, "name"):
        # This node has no mapping, it are potentially invalid
        return node.name + "?"
    return "?"


def reg(node: CustomNode):
    if isinstance(node, ConstructRegisterFromMetadataNode):
        return f"Register<4, {node.dtype}>"
    if hasattr(node, "type") and node.type is not None:
        return f"Register<4, {node.type.dtype}>"
    else:
        return f"Register<4, ?>"


@dataclass
class PlaceholderNode(CustomNode):
    _name: str
    type: Optional[DataType]

    def custom_string(self, value_map: dict[str, str]) -> str:
        return f"{self.name}"

    @classmethod
    def from_fx_node(cls, node: fx.Node):
        return cls(node.graph, node.op, node.name, node.type)


@dataclass
class ReadNode(CustomNode):
    memory: Union[fx.Proxy, "AllocSharedNode"]
    elements_per_thread: Optional[Any] = None
    type: Optional[Type[Register]] = None

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        simt_shape = self.elements_per_thread
        memory = getNode(self.memory)
        memory_type: Memory = self.memory.type
        return f"{name} %{value_map[memory.name]} -> Register<{memory_type.symbolic_shape}, {memory_type.dtype}> -> Register<{simt_shape}, {memory_type.dtype}>, indexing: {indexing(self)}\n"


def indexing(node: CustomNode):
    return str(node.fx_node.meta["index"])


@dataclass
class TiledLoop(CustomNode):
    axis: IndexExpr
    init_args: Sequence[Any]
    subgraph_name: str
    implicit_captures: Sequence[fx.Proxy]

    def emit(self):
        return super().emit()


@dataclass
class WriteNode(CustomNode):
    register_: fx.Proxy
    memory: Union[fx.Proxy, "AllocSharedNode"]
    elements_per_thread: Optional[Any]

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        # TODO: remove this when fx.Proxy is wrapped with a similar interface
        memory = getNode(self.memory)
        memory_type: Memory = memory.type
        return f"{name} %{value_map[self.register_.name]}, %{value_map[memory.name]} : Memory<{memory_type.symbolic_shape}, {memory_type.dtype}>\n"


# Nodes modeling TKL operations emitted only during codegen


@dataclass
class AllocSharedNode(CustomNode):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    type: Memory

    def emit(self):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = self.graph.create_node(
            "call_function",
            target=self.op,
            args=arg_list,
            name="alloc_shared",
            kwargs={},
        )

    def custom_string(self, value_map: dict[str, str]) -> str:
        return f"alloc : Memory<{self.shape}, {self.dtype}, #shared>\n"


@dataclass
class BarrierNode(CustomNode):
    def emit(self):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = self.graph.create_node(
            "call_function",
            target=self.op,
            args=arg_list,
            name="barrier",
            kwargs={},
        )


@dataclass
class GetResultNode(CustomNode):
    value: fx.Node
    index: int

    def emit(self):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = self.graph.create_node(
            "call_function",
            target=self.op,
            args=arg_list,
            name="get_result",
            kwargs={},
        )


@dataclass
class ReadSharedNode(CustomNode):
    memory: Union[fx.Proxy, "AllocSharedNode"]
    elements_per_thread: Optional[Any] = None
    type: Optional[Type[Register]] = None

    def emit(self):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = self.graph.create_node(
            "call_function",
            target=self.op,
            args=arg_list,
            name="read_shared",
            kwargs={},
        )

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        simt_shape = self.elements_per_thread
        memory = getNode(self.memory)
        memory_type: Memory = memory.type
        return f"{name} %{value_map[memory.fx_node.name]} -> Register<{memory_type.symbolic_shape}, {memory_type.dtype}> -> Register<{simt_shape}, {memory_type.dtype}>, indexing: {indexing(self)}\n"


@dataclass
class WriteSharedNode(CustomNode):
    register_: fx.Proxy
    memory: Union[fx.Proxy, "AllocSharedNode"]
    elements_per_thread: Optional[Any]

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        memory = getNode(self.memory)
        memory_type: Memory = memory.type
        return f"{name} %{value_map[self.register_.name]}, %{value_map[memory.fx_node.name]} : Memory<{memory_type.symbolic_shape}, {memory_type.dtype}>\n"

    def emit(self):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = self.graph.create_node(
            "call_function",
            target=self.op,
            args=arg_list,
            name="write_shared",
            kwargs={},
        )


# TODO: Use a decorator to register these properly
nodeTypes["barrier"] = BarrierNode
nodeTypes["construct_register_from_metadata"] = ConstructRegisterFromMetadataNode
nodeTypes["get_result"] = GetResultNode
nodeTypes["mma"] = MmaNode
nodeTypes["read_shared"] = ReadSharedNode
nodeTypes["write_shared"] = WriteSharedNode
nodeTypes["read"] = ReadNode
nodeTypes["write"] = WriteNode
nodeTypes["alloc_shared"] = AllocSharedNode
nodeTypes["tiled_loop"] = TiledLoop
nodeTypes["placeholder"] = PlaceholderNode


def getNode(node: fx.Node) -> CustomNode:
    for name, nodeT in nodeTypes.items():
        # The fx_nodes have a suffix to the name depending on the number of occurrences
        # of similar nodes. We are only interested in the name prefix.
        if node.name.startswith(name):
            return nodeT.from_fx_node(node)
        if node.op == "placeholder":
            return PlaceholderNode.from_fx_node(node)
    return UnknownNode.from_fx_node(node)

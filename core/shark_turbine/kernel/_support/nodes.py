from abc import ABC
from calendar import c
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Type, TypedDict
import torch.fx as fx
from .regions import RegionGraph
from ..lang.functional_types import Memory
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

    graph: RegionGraph
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


@dataclass
class UnknownNode(CustomNode):
    args: Sequence[Any]
    kwargs: dict[Any, Any]

    @classmethod
    def from_fx_node(cls, node: fx.Node):
        return cls(node.graph, node.op, node.args, node.kwargs)


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
    lhs: fx.Proxy
    rhs: fx.Proxy
    acc: fx.Proxy

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        lhs_memory_type: Memory = self.lhs.meta["type"]
        rhs_memory_type: Memory = self.lhs.meta["type"]
        acc_memory_type: Memory = self.lhs.meta["type"]

        return f"{name} %{value_map[self.lhs.name]}, %{value_map[self.rhs.name]}, %{value_map[self.acc.name]} : Register<4, {lhs_memory_type.dtype}>, Register<4, {rhs_memory_type.dtype}> -> Register<4, {acc_memory_type.dtype}>\n"


@dataclass
class ReadNode(CustomNode):
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        simt_shape = self.elements_per_thread
        memory_type: Memory = self.memory.meta["type"]
        return f"{name} %{value_map[self.memory.name]} -> Register<{memory_type.symbolic_shape}, {memory_type.dtype}> -> Register<{simt_shape}, {memory_type.dtype}> // register sizes hardcoded\n"


@dataclass
class WriteNode(CustomNode):
    register_: fx.Proxy
    memory: fx.Proxy
    elements_per_thread: Optional[Any]

    def custom_string(self, value_map: dict[str, str]) -> str:
        name = get_node_name(self.__class__.__name__)
        memory_type: Memory = self.memory.meta["type"]
        return f"{name} %{value_map[self.register_.name]}, %{value_map[self.memory.name]} : Memory<{memory_type.symbolic_shape}, {memory_type.dtype}>\n"


# TODO: Use a decorator to register these properly
nodeTypes["construct_register_from_metadata"] = ConstructRegisterFromMetadataNode
nodeTypes["mma"] = MmaNode
nodeTypes["read"] = ReadNode
nodeTypes["write"] = WriteNode


def getNode(node: fx.Node) -> CustomNode:
    for name, nodeT in nodeTypes.items():
        # The fx_nodes have a suffix to the name depending on the number of occurrences
        # of similar nodes. We are only interested in the name prefix.
        if node.name.startswith(name):
            return nodeT.from_fx_node(node)
    return UnknownNode.from_fx_node(node)
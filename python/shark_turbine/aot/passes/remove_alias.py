from typing import Callable

import torch
from torch.fx import (
    GraphModule,
    Node,
)
from torch.fx.experimental import proxy_tensor
from torch.utils import _pytree as pytree
import operator as py_operator


def remove_unbind(gm: GraphModule) -> GraphModule:
    # Find all unbind nodes
    unbind_nodes = []
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.unbind.int:
            unbind_nodes.append(node)

    to_erase = []

    # Replace all unbind -> getitem chains with a index_select node
    for unbind in unbind_nodes:
        only_getitem = True
        for user in unbind.users:
            if user.op != "call_function":
                only_getitem = False
                continue
            if user.target != py_operator.getitem:
                only_getitem = False
                continue
        if not only_getitem:
            continue

        unbind_dim = 0
        if len(unbind.args) == 2:
            unbind_dim = unbind.args[1]

        for user in unbind.users:
            # Get the getitem indices
            index = user.args[1]
            with gm.graph.inserting_before(user):
                select = gm.graph.call_function(
                    torch.select,
                    (unbind.args[0], unbind_dim, index),
                )
                # Replace the getitem node with the index_select node
                user.replace_all_uses_with(select)

            # Delete the getitem
            to_erase.append(user)

        to_erase.append(unbind)

    # Erase all the getitem nodes
    for node in to_erase:
        gm.graph.erase_node(node)
    gm.recompile()

    return gm


def remove_alias(gm: GraphModule, *args) -> GraphModule:
    # Replace unbind -> getitem chains with index_select
    gm = remove_unbind(gm)
    return gm

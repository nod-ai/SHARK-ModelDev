import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from typing import List, Optional

from .decompositions import DEFAULT_DECOMPOSITIONS

# These decompositions don't exist in 2.1.0, but are required in newer versions.
if hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu"):
    DEFAULT_DECOMPOSITIONS.append(
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    )


def apply_decompositions(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompose_ops: Optional[List[torch._ops.OpOverload]] = None,
):
    if decompose_ops is None:
        return gm

    decompositions = get_decompositions(decompose_ops)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)

    return gm


def turbine_cpu_pass_pipeline(gm: torch.fx.GraphModule, example_inputs):
    decompose_ops = DEFAULT_DECOMPOSITIONS
    return apply_decompositions(gm, example_inputs, decompose_ops)  # type: ignore

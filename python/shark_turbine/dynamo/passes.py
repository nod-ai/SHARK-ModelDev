import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from typing import List

# default decompositions pulled from SHARK
DEFAULT_DECOMPOSITIONS = [
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
            torch.ops.aten.native_layer_norm,
            torch.ops.aten.masked_fill.Tensor,
            torch.ops.aten.masked_fill.Scalar,
]

# decompositions that aid us in handling nn.BatchNorm2d
BATCHNORM_DECOMPOSITIONS = [
    torch.ops.aten._native_batch_norm_legit_functional,
    torch.ops.aten.squeeze.dims,
]


def apply_decompositions(gm: torch.fx.GraphModule, example_inputs, decompose_ops: List[torch._ops.OpOverload] = None):
    if decompose_ops is None:
        return gm

    decompositions = get_decompositions(decompose_ops)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)

    return gm


def turbine_cpu_pass_pipeline(gm: torch.fx.GraphModule, example_inputs):
    decompose_ops = DEFAULT_DECOMPOSITIONS + BATCHNORM_DECOMPOSITIONS
    return apply_decompositions(gm, example_inputs, decompose_ops)
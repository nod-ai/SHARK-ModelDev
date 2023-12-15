import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch._decomp import register_decomposition
from torch.func import functionalize
from torch import Tensor
from typing import Dict, List, Optional, Tuple


aten = torch._ops.ops.aten

# scaled_dot_product_attention used to be decomposed in pre-autograd, given that
# it calls _scaled_dot_product_attention_math and
# _scaled_dot_product_attention_math only has a CompositeImplicitAutograd
# kernel. As a result it's decomposed into ops with finer granularity.
# However recent PRs (#103826 #105131) added new logic in
# scaled_dot_product_attention and now it calls
# _scaled_dot_product_flash_attention which contains a CPU kernel. This results
# in _scaled_dot_product_flash_attention showing up in torch.export().
# This decomposition ensures scaled_dot_product_attention is still decomposed
# the same way as before, i.e., going through
# _scaled_dot_product_attention_math. Notice that this decomp rule should be
# excluded by inductor.
@register_decomposition(aten._scaled_dot_product_flash_attention.default)
def scaled_dot_product_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int, Tensor, Tensor, Tensor]:
    dtype = query.dtype
    batchSize, num_head, qSize, headSize = (
        query.shape[0],
        query.shape[1],
        query.shape[2],
        query.shape[3],
    )

    torch._check(
        torch.is_floating_point(query) and dtype is not torch.half,
        lambda: f"query must be FP32, FP64, BF16 but got {query.dtype}",
    )
    torch._check(
        query.dim() == 4 and key.dim() == 4 and value.dim() == 4,
        lambda: f"q, k, v must be a 4 dimensional tensor, got {query.dim()}, {key.dim()}, {value.dim()}",
    )
    torch._check(
        dropout_p == 0.0, lambda: f"dropout probability must be zero, got {dropout_p}"
    )
    torch._check(
        query.shape[3] == value.shape[3] and key.shape[3] == value.shape[3],
        lambda: "q, k, v should have the same head size",
    )
    torch._check(
        return_debug_mask is False, lambda: "return_debug_mask is not supported."
    )

    logsumexp = torch.empty([batchSize, qSize, num_head, headSize], dtype=torch.float)
    cum_seq_q, cum_seq_k = torch.empty([], dtype=torch.long), torch.empty(
        [], dtype=torch.long
    )
    max_q, max_k = 0, 0
    philox_seed, philox_offset = torch.empty([], dtype=torch.long), torch.empty(
        [], dtype=torch.long
    )
    debug_attn_mask = torch.empty(
        [],
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad,
    )
    output, _ = aten._scaled_dot_product_attention_math.default(
        query, key, value, None, dropout_p, is_causal, None, scale=scale
    )
    # Why this change?
    # In pre-dispatch export scaled_dot_product_attention is executed via
    # * flash_attention.
    # flash_attention allocates output tensor as (N, L, H, E)
    #   it then transposes that to get (N, H, L, E) which is supposed to be the return
    # tensor dim for scaled_dot_product_attention
    # assume x: [N, H, L, E] is the output sdpa
    # In MHA code, this output is then permuted via (2, 0, 1, 3) to get
    # (L, N, H, E) dim tensor
    # x = x.permute(2, 0, 1, 3).contiguous() and the viewed via
    # x = x.view(L * N, H * E)
    # During pre autograd dispatch call to contiguous is not traced because
    # flash_attention output after the x.permute is already contiguous
    # on which the view is valid
    # However, during 2nd stage export, post-dispatch, we run _match variant
    # instead of flash* to get the decomposition. _match variant returns
    # x: [N, H, L, E] applying x.permute(2, 0, 1, 3) returns
    # x: [L, N, H, E] and without converting this to contiguous tensor
    # subsequent view is not valid and the export fails
    # solution is to maintain the return tensor view from the decomp to be
    # exactly same as *flash* variant.
    # flash variants output is contiguous as [N, L, H, E]
    # _match variant out is contiguous as [N, H, L, E]
    # out = out.transpose(1, 2).contiguous gets output as contiguous
    # in [N, L, H, E].
    # Subsrequent transpose(1, 2) then returns a view on which
    # aforementioned code snippet, as showm below, is valid
    # x = x.permute(2, 0, 1, 3).contiguous() and the viewed via
    # x = x.view(L * N, H * E)

    # Really the invariant you want to maintain is:
    # pre-dispatch op-output and its decomposed representation must
    # return tensor with same view and dims
    output = output.transpose(1, 2).contiguous(memory_format=torch.contiguous_format)
    return (
        output.transpose(1, 2),
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )

# default decompositions pulled from SHARK / torch._decomp
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
    torch.ops.aten.t,
    torch.ops.aten.addmm,
    # decompositions that aid us in handling nn.BatchNorm2d
    torch.ops.aten._native_batch_norm_legit_functional,
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten.squeeze.dims,
    # decompositions for miscellaneous ops that are not handled in torch-mlir but have available decompositions
    torch.ops.aten.soft_margin_loss,
    torch.ops.aten.im2col,
    torch.ops.aten._euclidean_dist,
    torch.ops.aten.index_copy,
    torch.ops.aten.index_copy_,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten.log_sigmoid_forward,
    torch.ops.aten.unsafe_split.Tensor,
    torch.ops.aten.binary_cross_entropy,
    torch.ops.aten.dot,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten._prelu_kernel,
    torch.ops.aten.full,
    torch.ops.aten._log_softmax,
    torch.ops.aten.nll_loss_forward,
    torch.ops.aten.nll_loss_backward,
    torch.ops.aten._to_copy,
    torch.ops.aten._log_softmax_backward_data,
    torch.ops.aten.lift_fresh_copy.default,
    torch.ops.aten._unsafe_index.Tensor,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
]


def apply_decompositions(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompose_ops: List[torch._ops.OpOverload] = None,
):
    if decompose_ops is None:
        return gm

    decompositions = get_decompositions(decompose_ops)
    print(decompositions)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)

    return gm


def turbine_cpu_pass_pipeline(gm: torch.fx.GraphModule, example_inputs):
    decompose_ops = DEFAULT_DECOMPOSITIONS
    return apply_decompositions(gm, example_inputs, decompose_ops)

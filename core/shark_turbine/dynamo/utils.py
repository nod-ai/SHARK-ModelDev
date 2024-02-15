import torch
from torch._prims_common.wrappers import out_wrapper
from torch._prims_common import (
    DeviceLikeType,
    TensorLikeType,
)
import torch._refs as _refs
from torch._decomp import get_decompositions, register_decomposition
from torch import Tensor
from typing import Dict, List, Tuple, Optional


# manually add decomposition to bypass the error that comes
# from VAE encode(inp).latent_dist.sample() failing to symbolically
# trace from torch fx.
# Expected Torch stable version: > 2.1.0
# diffusers side issue: https://github.com/huggingface/diffusers/issues/6239
# temporary Torch fix: https://github.com/pytorch/pytorch/issues/107170
@register_decomposition(torch.ops.aten.randn.generator)
@out_wrapper()
def randn_generator(
    *shape,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: Optional[torch.layout] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLikeType:
    # We should eventually support the generator overload.
    # However, if someone passes in a None generator explicitly,
    # we can jut fall back to randn.default
    if generator is None:
        return _refs.randn(
            *shape,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    return NotImplemented

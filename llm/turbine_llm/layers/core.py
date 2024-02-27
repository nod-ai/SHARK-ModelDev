# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn

from ..data import (
    InferenceTensor,
    Theta,
)

__all__ = [
    "LinearLayer",
    "RMSNormLayer",
    "ThetaLayer",
    "TokenEmbedding",
]


class ThetaLayer(nn.Module):
    "Base class for layers that derive parameters from a Theta object."

    def __init__(self, theta: Theta):
        super().__init__()
        self.theta = theta

    def theta_tensor(self, name: str) -> InferenceTensor:
        # TODO: We may need to do some bookkeeping here to ensure export
        # tracks all of these.
        return self.theta.tensor(name)


class LinearLayer(ThetaLayer):
    """Linear layer which computes:

    ```
    matmul(x, weight.T)
    ```

    Whether the weight is transposed as part of the calculation can be
    controlled with `transpose_weight=` (default true).
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        transpose_weight: bool = True,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.transpose_weight = transpose_weight

    def forward(self, x: torch.Tensor):
        return self.theta.ops.matmul(
            x, self.weight, transpose_rhs=self.transpose_weight
        )


class RMSNormLayer(ThetaLayer):
    """Computes the unbiased full RMS layer normalization."""

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        epsilon: float = 1e-6,
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        return self.theta.ops.rms_norm(x, self.weight, epsilon=self.epsilon)


class TokenEmbedding(ThetaLayer):
    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
    ):
        super().__init__(theta)
        self.weight = self.theta_tensor(weight_name)

    def forward(self, input: torch.Tensor):
        return self.theta.ops.embedding_lookup(input, self.weight)

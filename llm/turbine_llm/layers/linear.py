# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from .base import Theta, ThetaLayer


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

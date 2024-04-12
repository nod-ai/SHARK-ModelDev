# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from .base import Theta, ThetaLayer


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

# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Abstract layout structs describing various physical arrangements.

These are typically logical, planar layouts over some fundamental data types.
Concrete sub-classes implement any necessary physical to logical mapping.
"""

from abc import abstractmethod
from typing import Optional

import torch

from .data import QuantizedLayout


class BlockScaledLayout(QuantizedLayout):
    """Block-quantized representation which consists of a scale (`d`) per
    block in a higher precision type. This arrangement does not apply an offset,
    and is therefore in the family of symmetric quantization schemes.

    The dequantization formula:

    ```
    result = d.to(dtype) * qs.to(dtype)
    ```

    The inner-most dims will retain block structure. For example, if the
    block size is 32 and the original shape was NxK, then the component
    shapes would be:

    * `d`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`
    """

    def __init__(self, shape: list[int], d: torch.Tensor, qs: torch.Tensor):
        self._shape = shape
        self._d = d
        self._qs = qs

    @property
    def shape(self) -> list[int]:
        """The flattened shape of the logical (unblocked) result."""
        return self._shape

    @property
    def d(self) -> torch.Tensor:
        """Per block scales."""
        return self._d

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values."""
        return self._qs

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        qs = self.qs
        if dtype:
            d = d.to(dtype)
        else:
            dtype = d.dtype
        scaled = d * qs.to(dtype)
        return scaled

    def __repr__(self):
        return (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"qs({list(self.qs.shape)}, dtype={self.qs.dtype}))"
        )

# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Abstract layout structs describing various physical arrangements.

These are typically logical, planar layouts over some fundamental data types.
Concrete sub-classes implement any necessary physical to logical mapping.
"""

from typing import Optional

import torch

from .tensors import QuantizedLayout

from .layout_utils import promote_linear_i4_block_to_i8

__all__ = [
    "BlockScaledI4Layout",
    "BlockScaledLayout",
]


class BlockScaledLayout(QuantizedLayout):
    """Block-quantized representation which consists of a scale (`d`)
    and offset (`m`) per block in a higher precision type. The offset, if
    present, is pre-scaled.

    The dequantization formula:

    ```
    result = d.to(dtype) * qs.to(dtype) + m.to(dtype)
    ```

    The inner-most dims will retain block structure. For example, if the
    block size is 32 and the original shape was NxK, then the component
    shapes would be:

    * `d`: `[N, K // 32, 1]`
    * `m`: `[N, K // 32, 1]`
    * `qs`: `[N, K // 32, 32]`

    Note that the offset (`m`) is optional.
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        *,
        m: Optional[torch.Tensor] = None,
    ):
        self._shape = shape
        self._d = d
        self._m = m
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
    def m(self) -> torch.Tensor:
        """Per block offsets."""
        return self._m

    @property
    def qs(self) -> torch.Tensor:
        """Per sample quantized values."""
        return self._qs

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        m = self.m
        qs = self.qs
        if dtype:
            d = d.to(dtype)
            if m is not None:
                m = m.to(dtype)
        else:
            dtype = d.dtype
            assert m is None or m.dtype == d.dtype
        scaled = d * qs.to(dtype)
        shifted = scaled if m is None else scaled + m
        return shifted

    def __repr__(self):
        r = (
            f"{type(self).__name__}(d({list(self.d.shape)}, dtype={self.d.dtype}), "
            f"qs({list(self.qs.shape)}, dtype={self.qs.dtype}))"
        )
        if self.m is not None:
            r += f", m({list(self.m.shape)}, dtype={self.m.dtype})"
        return r


class BlockScaledI4Layout(BlockScaledLayout):
    """A BlockScaledLayout where the `qs` are internally packed 2 values per byte.

    Per convention, the `qs` property returns a tensor as either uint8 or
    int8 (depending on `signed=`) that can be used directly for arithmetic.
    The underlying bit-packed tensor can be accessed via `qs_bit_packed` and
    it is laid out in little endian bit order, linearly across the block
    dimension. There are an arbitrary ways to organize such things, and
    if more specificity is needed, a dedicated layout class should be used. In
    general, for these "generic" layouts, we choose defaults that mate well
    with how the compiler infra and prevailing targets are built and trust that
    optimizations that care will choose a specific packing.
    """

    def __init__(
        self,
        shape: list[int],
        d: torch.Tensor,
        qs: torch.Tensor,
        *,
        m: Optional[torch.Tensor] = None,
        signed: bool = False,
    ):
        super().__init__(shape, d, qs, m=m)
        self.signed = signed

    @property
    def qs(self) -> torch.Tensor:
        # `qs` is defined as something that we can do integer arithmetic on
        # for cases where we only have non-packed kernels available. Therefore,
        # we promote it to i8. The `qs_packed` is available for the sub-byte
        # bit pattern.
        return promote_linear_i4_block_to_i8(self._qs, signed=self.signed)

    @property
    def qs_bit_packed(self) -> torch.Tensor:
        return self._qs

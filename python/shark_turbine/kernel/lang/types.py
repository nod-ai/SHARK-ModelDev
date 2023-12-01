import torch

__all__ = [
    "KernelBuffer",
    "Grid",
]

Grid = tuple[int, ...]


class KernelBuffer:
    """Represents a buffer in global memory.

    Top level kernels always operate on global memory via these
    buffers, and the primary operations that can be performed on
    them are loads/stores and DMAs to some form of compute
    capable local buffer.

    When executing eagerly, these are backed by a normal torch
    Tensor. When compiling, an appropriate duck-typed proxy
    is used.
    """

    __slots__ = [
        "_tensor",
    ]

    def __init__(self, tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor but got {tensor}"
        self._tensor = tensor

    def __repr__(self):
        return f"KernelBuffer({self._tensor})"

    def __setitem__(self, key, item):
        self._tensor.__setitem__(key, item)

    def __getitem__(self, key):
        return self._tensor.__getitem__(key)

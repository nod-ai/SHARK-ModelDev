# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..layers import *
from ..types import *


def main():
    from ..utils import cli

    parser = cli.create_parser()
    cli.add_gguf_dataset_options(parser)
    args = cli.parse(parser)

    data_files = cli.get_gguf_data_files(args)
    config = gguf_interop.load_file(data_files["gguf"])

    print(f"Properties:")
    for key, value in config.properties.items():
        print(f"  {key} = {value} (of {type(value)})")
    print("Tensors:")
    for tensor in config.root_theta.flatten().values():
        print(f"  {tensor}")
        if isinstance(tensor, PrimitiveTensor):
            torch_tensor = tensor.as_torch()
            print(
                f"  : torch.Tensor({list(torch_tensor.shape)}, "
                f"dtype={torch_tensor.dtype}) = {tensor.as_torch()}"
            )
        else:
            assert isinstance(tensor, QuantizedTensor), f"Got {type(tensor)}"
            raw = tensor.raw  # type: ignore
            print(
                f"  : QuantizedTensor({tensor.layout_type.__name__})="
                f"torch.Tensor({list(raw.shape)}, dtype={raw.dtype})"
            )
            try:
                unpacked = tensor.unpack()
                print(f"    {unpacked}")
            except NotImplementedError:
                print(f"     NOT IMPLEMENTED")


if __name__ == "__main__":
    main()

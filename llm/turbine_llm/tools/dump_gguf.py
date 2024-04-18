# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import re

import numpy as np
import torch

from ..layers import *
from ..types import *


def main():
    from ..utils import cli

    parser = cli.create_parser()
    cli.add_gguf_dataset_options(parser)
    parser.add_argument(
        "--dump-tensor-dir", type=Path, help="Dump tensor contents to a directory"
    )
    parser.add_argument(
        "--tensor-regex", type=str, help="Only dumps tensors matching a regex"
    )
    args = cli.parse(parser)

    data_files = cli.get_gguf_data_files(args)
    config = gguf_interop.load_file(data_files["gguf"])

    print(f"Properties:")
    for key, value in config.properties.items():
        print(f"  {key} = {value} (of {type(value)})")
    print("Tensors:")
    for tensor in config.root_theta.flatten().values():
        if args.tensor_regex is not None:
            if not re.search(args.tensor_regex, tensor.name):
                continue
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

        _maybe_dump_tensor(args, tensor)


def _maybe_dump_tensor(args, t: InferenceTensor):
    if not args.dump_tensor_dir:
        return
    dir: Path = args.dump_tensor_dir
    dir.mkdir(parents=True, exist_ok=True)
    print(f"    (Dumping to {dir})")

    if isinstance(t, PrimitiveTensor):
        torch_tensor = t.as_torch()
        np.save(dir / f"{t.name}.npy", torch_tensor.detach().numpy())
    elif isinstance(t, QuantizedTensor):
        layout: QuantizedLayout = t.unpack()
        dq = layout.dequant()
        np.save(dir / f"{t.name}.dequant.npy", dq.detach().numpy())
    else:
        raise AssertionError(f"Unexpected tensor type: {type(t)}")


if __name__ == "__main__":
    main()

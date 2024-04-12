# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Contains utilities for fetching datasets from huggingface.

There is nothing special about this mechanism, but it gives us a common
place to stash dataset information for testing and examples.

This can be invoked as a tool in order to fetch a local dataset.
"""

from typing import Dict, Optional, Sequence, Tuple

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


################################################################################
# Dataset support
################################################################################


@dataclass
class RemoteFile:
    file_id: str
    repo_id: str
    filename: str
    extra_filenames: Sequence[str] = ()

    def download(self, *, local_dir: Optional[Path] = None) -> Path:
        for extra_filename in self.extra_filenames:
            hf_hub_download(
                repo_id=self.repo_id, filename=extra_filename, local_dir=local_dir
            )
        return Path(
            hf_hub_download(
                repo_id=self.repo_id, filename=self.filename, local_dir=local_dir
            )
        )


@dataclass
class Dataset:
    name: str
    files: Tuple[RemoteFile]

    def __post_init__(self):
        if self.name in ALL_DATASETS:
            raise KeyError(f"Duplicate dataset name '{self.name}'")
        ALL_DATASETS[self.name] = self

    def alias_to(self, to_name: str) -> "Dataset":
        alias_dataset(self.name, to_name)
        return self

    def download(self, *, local_dir: Optional[Path] = None) -> Dict[str, Path]:
        return {f.file_id: f.download(local_dir=local_dir) for f in self.files}


ALL_DATASETS: Dict[str, Dataset] = {}


def get_dataset(name: str) -> Dataset:
    try:
        return ALL_DATASETS[name]
    except KeyError:
        raise KeyError(f"Dataset {name} not found (available: {ALL_DATASETS.keys()})")


def alias_dataset(from_name: str, to_name: str):
    if to_name in ALL_DATASETS:
        raise KeyError(f"Cannot alias dataset: {to_name} already exists")
    ALL_DATASETS[to_name] = get_dataset(from_name)


################################################################################
# Dataset definitions
################################################################################

Dataset(
    "SlyEcho/open_llama_3b_v2_f16_gguf",
    (
        RemoteFile(
            "gguf", "SlyEcho/open_llama_3b_v2_gguf", "open-llama-3b-v2-f16.gguf"
        ),
        RemoteFile(
            "tokenizer_config.json",
            "openlm-research/open_llama_3b_v2",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("open_llama_3b_v2_f16_gguf")


################################################################################
# Tool entrypoint
################################################################################


def main():
    parser = argparse.ArgumentParser("hf_datasets")
    parser.add_argument(
        "dataset_name",
        nargs="+",
        help=f"Dataset to request (available = {list(ALL_DATASETS.keys())})",
    )
    parser.add_argument(
        "--local-dir", type=Path, help="Link all files to a local directory"
    )
    args = parser.parse_args()

    if args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.dataset_name:
        print(f"Downloading dataset {dataset_name}")
        ds = get_dataset(dataset_name).download(local_dir=args.local_dir)
        for key, path in ds.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()

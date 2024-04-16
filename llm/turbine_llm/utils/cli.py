# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Utilities for building command line tools."""

from typing import Dict, Optional

import argparse
from pathlib import Path

from . import hf_datasets
from . import tokenizer


def create_parser(
    *,
    prog: Optional[str] = None,
    usage: Optional[str] = None,
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, usage=usage, description=description)

    return parser


def parse(parser: argparse.ArgumentParser):
    """Parses arguments and does any prescribed global process setup."""
    return parser.parse_args()


def add_gguf_dataset_options(parser: argparse.ArgumentParser):
    """Adds options to load a GGUF dataset.

    Either the `--hf-dataset` or `--gguf-file` argument can be present.
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hf-dataset",
        help=f"HF dataset to use (available: {list(hf_datasets.ALL_DATASETS.keys())})",
    )
    group.add_argument("--gguf-file", type=Path, help="GGUF file to load")


def add_tokenizer_options(parser: argparse.ArgumentParser):
    """Adds options for specifying a tokenizer.

    All are optional and if not specified, some default options will be taken
    based on the dataset.
    """
    parser.add_argument(
        "--tokenizer-type", help="Tokenizer type or infer from dataset if not specified"
    )
    parser.add_argument(
        "--tokenizer-config-json",
        help="Direct path to a tokenizer_config.json file",
        type=Path,
    )


def get_gguf_data_files(args) -> Dict[str, Path]:
    """Gets the path to the gguf file for the dataset."""
    if args.hf_dataset is not None:
        dataset = hf_datasets.get_dataset(args.hf_dataset).download()
        if "gguf" not in dataset:
            raise ValueError(
                f"Argument --hf-dataset refers to a dataset that does not contain a "
                f"gguf file ({dataset})"
            )
        return dataset
    else:
        return {"gguf": args.gguf_file}


def get_tokenizer(
    args, *, data_files: Optional[Dict[str, Path]] = None
) -> tokenizer.InferenceTokenizer:
    """Gets a tokenizer based on arguments.

    If the data_files= dict is present and explicit tokenizer options are not
    set, we will try to infer a tokenizer from the data files.
    """
    if data_files is None:
        data_files = {}
    else:
        data_files = dict(data_files)
    if args.tokenizer_config_json is not None:
        data_files["tokenizer_config.json"] = args.tokenizer_config_json

    tokenizer_type = args.tokenizer_type
    if tokenizer_type is None:
        if "tokenizer_config.json" in data_files:
            return tokenizer.load_tokenizer(
                data_files["tokenizer_config.json"].parent,
                tokenizer_type="transformers",
            )
        else:
            raise ValueError(f"Could not infer tokenizer from data files: {data_files}")
    else:
        raise ValueError(f"Unsupported --tokenizer-type argument: {tokenizer_type}")

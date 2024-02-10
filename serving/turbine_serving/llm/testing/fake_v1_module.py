# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implements a service_v1 compliant module in Python for testing.

This uses a PyModuleInterface to define a fake VmModule that exposes 'prefill_bs{n}'
and 'decode_bs{n}' such that the call sequence and args/results can be manipulated.
"""

from iree.runtime import (  # type: ignore
    HalBuffer,
    HalBufferView,
    HalElementType,
    PyModuleInterface,
    VmModule,
)

from ..config import ModelParams


def create_fake_module(module_name: str, model_params: ModelParams) -> VmModule:
    class ServiceV1Module:
        def __init__(self, iface):
            ...

        def prefill(
            self,
            bs: int,
            # token_ids_ref,
            # seq_lens_ref,
            attn_block_bv_ref,
            tied_attn_block_buffer_ref_inp,
            tied_attn_block_buffer_ref_out,
        ):
            attn_block_bv = attn_block_bv_ref.deref(HalBufferView)
            tied_attn_block_buffer = tied_attn_block_buffer_ref_out.deref(HalBuffer)
            print(f"FAKE_V1_MODULE: PREFILL bs={bs}")
            print(f"  attn_block (input): {attn_block_bv}")
            print(f"  TIED RESULT attn_block_buffer: {tied_attn_block_buffer}")

            return attn_block_bv_ref

        def decode(self, bs: int):
            print(f"FAKE_V1_MODULE: DECODE bs={bs}")

    iface = PyModuleInterface(module_name=module_name, ctor=ServiceV1Module)

    # Dynamically define prefill functions.
    def add_prefill_bs(bs: int):
        def trampoline(self, *args):
            return self.prefill(bs, *args)

        iface.export(f"prefill_bs{bs}", "0rrr_r", trampoline)

    [add_prefill_bs(bs) for bs in model_params.prefill_batch_sizes]

    # Dynamically define decode functions.
    def add_decode_bs(bs: int):
        def trampoline(self, *args):
            return self.decode(bs, *args)

        iface.export(f"decode_bs{bs}", "0v_v", trampoline)

    [add_decode_bs(bs) for bs in model_params.decode_batch_sizes]

    return iface.create()

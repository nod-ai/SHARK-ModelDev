# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implements a service_v1 compliant module in Python for testing.

This uses a PyModuleInterface to define a fake VmModule that exposes 'prefill_bs{n}'
and 'decode_bs{n}' such that the call sequence and args/results can be manipulated.
"""

import textwrap
import threading

from iree.runtime import (  # type: ignore
    HalBuffer,
    HalBufferView,
    HalElementType,
    HalFence,
    PyModuleInterface,
    VmModule,
    VmRef,
)

from ..config import ModelParams


def create_fake_module(module_name: str, model_params: ModelParams) -> VmModule:
    class ServiceV1Module:
        def __init__(self, iface):
            ...

        def prefill(
            self,
            bs: int,
            token_ids_ref: VmRef,
            seq_lens_ref: VmRef,
            attn_block_indices_ref: VmRef,
            attn_block_bv_ref: VmRef,
            tied_attn_block_buffer_ref_inp: VmRef,
            tied_attn_block_buffer_ref_out: VmRef,
            wait_fence_ref: VmRef,
            signal_fence_ref: VmRef,
        ):
            def run():
                print(f"FAKE_V1_MODULE: PREFILL bs={bs} : WAIT")
                wait_fence = wait_fence_ref.deref(HalFence)  # type: HalFence
                signal_fence = signal_fence_ref.deref(HalFence)  # type: HalFence
                try:
                    wait_fence.wait()
                    print("  - READY")

                    tied_attn_block_buffer = tied_attn_block_buffer_ref_out.deref(
                        HalBuffer
                    )
                    _format_device_buffer_view(
                        lambda s: print("  token_ids =", s), token_ids_ref
                    )
                    _format_device_buffer_view(
                        lambda s: print("  seq_lens =", s), seq_lens_ref
                    )
                    _format_device_buffer_view(
                        lambda s: print("  attn_block_indices =", s),
                        attn_block_indices_ref,
                    )
                    _format_device_buffer_view(
                        lambda s: print("  attn_block =", s), attn_block_bv_ref
                    )

                    print(f"  TIED RESULT attn_block_buffer: {tied_attn_block_buffer}")
                    signal_fence.signal()
                except Exception as e:
                    signal_fence.fail(str(e))

            threading.Thread(target=run).start()
            return attn_block_bv_ref

        def decode(self, bs: int):
            print(f"FAKE_V1_MODULE: DECODE bs={bs}")

    iface = PyModuleInterface(module_name=module_name, ctor=ServiceV1Module)

    # Dynamically define prefill functions.
    def add_prefill_bs(bs: int):
        def trampoline(self, *args):
            return self.prefill(bs, *args)

        iface.export(f"prefill_bs{bs}", "0rrrrrrrr_r", trampoline)

    [add_prefill_bs(bs) for bs in model_params.prefill_batch_sizes]

    # Dynamically define decode functions.
    def add_decode_bs(bs: int):
        def trampoline(self, *args):
            return self.decode(bs, *args)

        iface.export(f"decode_bs{bs}", "0v_v", trampoline)

    [add_decode_bs(bs) for bs in model_params.decode_batch_sizes]

    return iface.create()


def _format_device_buffer_view(callback, bv_ref: VmRef):
    bv = bv_ref.deref(HalBufferView)  # type: HalBufferView
    value = bv.map().asarray(bv.shape, HalElementType.map_to_dtype(bv.element_type))
    value_indented = textwrap.indent(repr(value), "    ")
    callback(f"{bv!r}\n{value_indented}")

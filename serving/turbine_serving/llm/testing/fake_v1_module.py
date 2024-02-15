# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Implements a service_v1 compliant module in Python for testing.

This uses a PyModuleInterface to define a fake VmModule that exposes 'prefill_bs{n}'
and 'decode_bs{n}' such that the call sequence and args/results can be manipulated.
"""

import numpy as np
import textwrap
import threading

from iree.runtime import (  # type: ignore
    BufferUsage,
    HalBuffer,
    HalBufferView,
    HalDevice,
    HalElementType,
    HalFence,
    MemoryType,
    PyModuleInterface,
    VmModule,
    VmRef,
)

from ..config import ModelParams


def create_fake_module(
    device: HalDevice, module_name: str, model_params: ModelParams
) -> VmModule:
    class ServiceV1Module:
        def __init__(self, iface):
            ...
            print("IFACE:", iface, dir(iface))

        def prefill(
            self,
            bs: int,
            token_ids_ref: VmRef,
            seq_lens_ref: VmRef,
            attn_block_indices_ref: VmRef,
            wait_fence_ref: VmRef,
            signal_fence_ref: VmRef,
        ):
            result_array: np.ndarray = np.ndarray([bs, 1], dtype=np.int32)

            def run():
                print(f"FAKE_V1_MODULE: PREFILL bs={bs} : WAIT")
                wait_fence = wait_fence_ref.deref(HalFence)  # type: HalFence
                signal_fence = signal_fence_ref.deref(HalFence)  # type: HalFence
                try:
                    wait_fence.wait()
                    print("  - READY")
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

                    # Async populate.
                    device_array = result_bv.map().asarray(
                        result_array.shape, result_array.dtype
                    )
                    for i in range(bs):
                        device_array[i, 0] = i + 1

                    signal_fence.signal()
                except Exception as e:
                    signal_fence.fail(str(e))

            threading.Thread(target=run).start()

            result_buffer = device.allocator.allocate_buffer(
                memory_type=MemoryType.DEVICE_LOCAL | MemoryType.HOST_VISIBLE,
                allowed_usage=BufferUsage.DEFAULT,
                allocation_size=result_array.size * result_array.itemsize,
            )
            result_bv = HalBufferView(
                result_buffer, result_array.shape, HalElementType.INT_32
            )
            return result_bv.ref

        def decode(self, bs: int):
            print(f"FAKE_V1_MODULE: DECODE bs={bs}")

    iface = PyModuleInterface(module_name=module_name, ctor=ServiceV1Module)

    # Dynamically define prefill functions.
    def add_prefill_bs(bs: int):
        def trampoline(self, *args):
            return self.prefill(bs, *args)

        iface.export(f"prefill_bs{bs}", "0rrrrr_r", trampoline)

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

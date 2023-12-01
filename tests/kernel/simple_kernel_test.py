# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch

import shark_turbine.kernel as tk


class Test(unittest.TestCase):
    def testIotaEager(self):
        @tk.gen.thread
        def iota_kernel(out: tk.lang.GlobalBuffer):
            i = tk.lang.program_id(0)
            out[i] = i

        print("iota_kernel:", iota_kernel)
        print("iota_kernel[8]:", iota_kernel[8])
        print("iota_kernel[8, 1]:", iota_kernel[8, 1])
        out = torch.empty(8, dtype=torch.int32)
        iota_kernel[8](out)
        print(out)

    def testIotaFx(self):
        @tk.gen.thread
        def iota_kernel(out: tk.lang.GlobalBuffer):
            i = tk.lang.program_id(0)
            out[i] = i

        print(iota_kernel._trace.gm.graph)
        # Prints:
        # .graph():
        #     %out : shark_turbine.kernel.lang.types.GlobalBuffer [num_users=1] = placeholder[target=out]
        #     %program_id : [num_users=1] = call_function[target=shark_turbine.kernel.lang.prims.program_id](args = (0,), kwargs = {})
        #     %_global_buffer_setitem : [num_users=0] = call_function[target=shark_turbine.kernel._support.tracing._global_buffer_setitem](args = (%out, %program_id, %program_id), kwargs = {})
        #     return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

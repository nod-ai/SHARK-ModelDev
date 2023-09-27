# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch._dynamo as dynamo

import shark_turbine.kernel as tk


class Test(unittest.TestCase):
    def testIotaEager(self):
        @tk.block_kernel(eager=True)
        def iota_kernel(out: tk.KernelBuffer):
            i = tk.program_id(0)
            out[i] = i

        gridded = iota_kernel(grid=(8,))
        out = torch.empty(8, dtype=torch.int32)
        gridded(out)
        print(out)

    def testIotaFx(self):
        @tk.block_kernel
        def iota_kernel(out: tk.KernelBuffer):
            i = tk.program_id(0)
            out[i] = i

        print(iota_kernel.tk_trace.gm.graph)
        # Prints:
        # .graph():
        #     %out : shark_turbine.kernel.core.KernelBuffer [num_users=1] = placeholder[target=out]
        #     %program_id : [num_users=1] = call_function[target=shark_turbine.kernel.core.program_id](args = (0,), kwargs = {})
        #     %_kernel_buffer_setitem : [num_users=0] = call_function[target=shark_turbine.kernel.core._kernel_buffer_setitem](args = (%out, %program_id, %program_id), kwargs = {})
        #     return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

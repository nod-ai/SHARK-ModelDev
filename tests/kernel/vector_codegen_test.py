import logging
import unittest

import torch
import shark_turbine.kernel as tk

from shark_turbine.kernel.compiler import (
    builder,
    vector_codegen,
)
from shark_turbine.kernel._support import (
    indexing,
)

M = tk.lang.sym.M
K = tk.lang.sym.K


class Test(unittest.TestCase):
    # This test is using the compiler "the hard way" until we have all of the
    # API layering in place.
    def testIotaFx(self):
        @tk.gen.thread(M)
        def iota_kernel(out: tk.lang.OutputBuffer[M]):
            i = tk.lang.program_id(0)
            secret_value = ((i * (33 - i) + 4) % 8) // 2
            out[i] = secret_value

        gm = iota_kernel._trace.gm
        print(gm.graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_constant(M, 17)

            sig = vector_codegen.Signature()
            sig.add_from_graph_placeholders(gm.graph)
            sig.add_grid(iota_kernel.grid_type)
            print(sig)
            try:
                emitter = vector_codegen.ThreadEmitter(mb, iota_kernel.grid_type, sig)
                emitter.emit_graph(gm.graph)
                emitter.finish()
            finally:
                print(mb.module_op.get_asm())
            mb.module_op.verify()

    def testSoftmaxFx(self):
        @tk.gen.thread(M)
        def softmax_kernel(
            input: tk.lang.KernelBuffer[M, K], output: tk.lang.KernelBuffer[M, K]
        ):
            row_index = tk.lang.program_id(0)
            input_row = input[row_index, :]
            numerator = torch.exp(input_row - torch.max(input_row))
            output_row = numerator / torch.sum(numerator)
            output[row_index, :] = output_row

        gm = softmax_kernel._trace.gm
        print(gm.graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_constant(M, 17)

            sig = vector_codegen.Signature()
            sig.add_from_graph_placeholders(gm.graph)
            sig.add_grid(softmax_kernel.grid_type)
            print(sig)
            try:
                emitter = vector_codegen.ThreadEmitter(
                    mb, softmax_kernel.grid_type, sig
                )
                emitter.emit_graph(gm.graph)
                emitter.finish()
            finally:
                print(mb.module_op.get_asm())
            mb.module_op.verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

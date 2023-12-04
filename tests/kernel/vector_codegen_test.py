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
            out[i] = i

        gm = iota_kernel._trace.gm
        print(gm.graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_constant(M, 17)

            sig = vector_codegen.Signature()
            sig.add_from_graph_placeholders(gm.graph)
            sig.add_grid(iota_kernel.grid_type)
            print(sig)
            emitter = vector_codegen.ThreadEmitter(mb, iota_kernel.grid_type, sig)
            emitter.emit_graph(gm.graph)
            emitter.finish()
            print(mb.module_op.get_asm())
            mb.module_op.verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

import logging
import unittest
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf

class Test(unittest.TestCase):
    def testGemm(self):
        # Tensor dimensions
        M = tkl.sym.M
        N = tkl.sym.N
        K = tkl.sym.K
        # Address space (for GPU, shared or global)
        S = tkl.sym.S

        # This example hardcodes the hardware and distribution constraints,
        # but these should be exposed in the language.

        # Warp-level micro-kernel (inner-most loop).
        # Since warps are not directly addressable, there is no
        # explicit notion of a warp id (like a workgroup or thread id).
        # Here we use a functional style of expressing the loop since
        # we do not know the loop bounds.
        @tkf.wave(0, 0)
        def gemm(a: tkf.Memory[M, K, S, tkl.f16],
                 b: tkf.Memory[N, K, S, tkl.f16],
                 creg: tkf.Register[M, N, tkl.f32]) -> tkf.Register[M, N, tkl.f32]:
            areg = tkf.memory_to_register(a)
            breg = tkf.memory_to_register(b)
            dreg = tkf.mma(areg, breg, creg)
            return dreg


        with tk.gen.TestLaunchContext():
            # Inputs
            a = torch.randn(64, 128, dtype=torch.float16)
            b = torch.randn(256, 128, dtype=torch.float16)
            # Outputs
            c = torch.zeros(64, 256, dtype=torch.float32)
            gemm(a, b, c)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
import logging
import unittest
import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.functional as tkf


class Test(unittest.TestCase):
    def testGemm(self):

        # Wave tile sizes (determined by constraints below)
        M = tkl.sym.M
        N = tkl.sym.N
        K = tkl.sym.K

        # Workgroup tile sizes
        BLOCK_M = tkl.sym.BLOCK_M
        BLOCK_N = tkl.sym.BLOCK_N
        BLOCK_K = tkl.sym.BLOCK_K
        # Address space (for GPU, shared(1) or global(0))
        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
        # Other hyperparameters
        LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
        STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

        # Expose user-constraints
        constraints = [tkf.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkf.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkf.TilingConstraint(K, BLOCK_K)]
        constraints += [tkf.ThreadConstraint(threads_per_block=[64, 1, 1])]
        constraints += [
            tkf.HardwareConstraint(
                threads_per_wave=64, mma_type="MFMA_F32_16x16x16_F16"
            )
        ]

        # Wave-level micro-kernel.
        # Since warps are not directly addressable, there is no
        # explicit notion of a warp id (like a workgroup or thread id).
        # Here we use a functional style of expressing the loop since
        # we do not know the loop bounds.
        @tkf.wave(constraints)
        def gemm(
            a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
            c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        ):
            # This microkernel encodes the fact that if the reduction
            # dimension were tiled, then we would need to materialize a loop.
            # c_reg: tkf.Register[WAVE_M, WAVE_N, tkl.f32]
            c_reg = tkf.construct_register_from_metadata((M, N), tkl.f32, 0.0)

            # Do we maybe rather need the info that this is a reduction dimension?
            # This could be called tkf.dim(K) or tkf.reduction(K) ?
            @tkf.tiled_loop(K, init_args=[c_reg])
            def repeat(c_reg) -> tkl.Register[M, N, tkl.f32]:
                # a_reg: tkf.Register[M, K, tkl.f16]
                # b_reg: tkf.Register[N, K, tkl.f16]
                a_reg = tkf.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                b_reg = tkf.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                c_reg = tkf.mma(a_reg, b_reg, c_reg)
                return c_reg

            # Call removed as the init arg is now explicit above.
            # result = repeat(c_reg)
            tkf.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)
            # We also discussed using `repeat` directly in tkf.write:
            # tkf.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

        hyperparams = {
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 1,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
            M: 64,
            N: 64,
            K: 256,
        }
        with tk.gen.TestLaunchContext(hyperparams):
            a = torch.randn(64, 256, dtype=torch.float16)
            b = torch.randn(64, 256, dtype=torch.float16)
            c = torch.zeros(64, 64, dtype=torch.float32)
            gemm(a, b, c)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

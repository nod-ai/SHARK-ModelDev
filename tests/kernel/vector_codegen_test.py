import logging
import unittest

import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl

from shark_turbine.kernel.compiler import (
    builder,
    kernel_codegen,
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

        trace = iota_kernel._trace
        print(trace.region_graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_constant(M, 17)
            idxc.finalize()
            sig = kernel_codegen.KernelSignature()
            sig.add_from_graph_placeholders(trace.get_root_graph())
            sig.add_grid(iota_kernel.grid_type)
            print(sig)
            bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(
                sig, mb
            )
            try:
                emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
                emitter.emit()
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

        trace = softmax_kernel._trace
        print(trace.region_graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_constant(M, 128)
            idxc.bind_constant(K, 64)
            idxc.finalize()

            sig = kernel_codegen.KernelSignature()
            sig.add_from_graph_placeholders(trace.get_root_graph())
            sig.add_grid(softmax_kernel.grid_type)
            print(sig)
            bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(
                sig, mb
            )
            emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
            try:
                emitter.emit()
            finally:
                emitter.finish()
                print(mb.module_op.get_asm())
            mb.module_op.verify()

    def testForLoopFx(self):
        @tk.gen.thread(M)
        def for_loop_kernel(
            input: tk.lang.KernelBuffer[M, K], output: tk.lang.KernelBuffer[M, K]
        ):
            row_idx = tkl.program_id(0)
            sum = input[row_idx, 0]
            prefetch = input[row_idx, 1]

            @tkl.for_loop(2, 5, init_args=[sum, prefetch])
            def prefetch_sum(i, iter_args):
                new_sum = iter_args[0] + iter_args[1]
                new_prefetch = input[row_idx, i]
                return new_sum, new_prefetch

            output[row_idx, 0] = prefetch_sum[0]

        trace = for_loop_kernel._trace
        print(trace.region_graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_constant(M, 128)
            idxc.bind_constant(K, 64)
            idxc.finalize()

            sig = kernel_codegen.KernelSignature()
            sig.add_from_graph_placeholders(trace.get_root_graph())
            sig.add_grid(for_loop_kernel.grid_type)
            print(sig)
            bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(
                sig, mb
            )
            emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
            try:
                emitter.emit()
            finally:
                emitter.finish()
                print(mb.module_op.get_asm())
            mb.module_op.verify()

    def testGemmFx(self):
        N = tkl.sym.N
        M = tkl.sym.M
        K = tkl.sym.K

        GRID_N = tkl.sym.GRID_N
        GRID_M = tkl.sym.GRID_M

        def inner_gemm(
            A: tkl.KernelBuffer[N, K],
            B: tkl.KernelBuffer[K, M],
            output: tkl.KernelBuffer[N, M],
            k: int,
            block_size: int,
        ):
            grid_n = tkl.program_id(0)
            grid_m = tkl.program_id(1)

            acc = tkl.constant((block_size, block_size), torch.float32, 0.0)

            @tkl.for_loop(0, k // block_size, init_args=[acc])
            def body(i, c):
                a = tkl.load(A, (grid_n, i * block_size), (block_size, block_size))
                b = tkl.load(B, (i * block_size, grid_m), (block_size, block_size))
                return (tkl.dot(a, b, c),)

            tkl.store(output, (grid_n, grid_m), body[0])

        @tk.gen.thread(GRID_N, GRID_M)
        def gemm_kernel(
            A: tkl.KernelBuffer[N, K],
            B: tkl.KernelBuffer[K, M],
            output: tkl.KernelBuffer[N, M],
        ):
            # TODO: We should find a way to parameterize these so we can autotune over them.
            # TODO: Ideally, we should be getting k from the symbol. The symbol value
            # is currently not available at tracing time which is a problem.
            k = 512
            block_size = 32
            inner_gemm(A, B, output, k, block_size)

        trace = gemm_kernel._trace
        print(trace.region_graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            BLOCK_SIZE = 32
            idxc.bind_constant(N, 512)
            idxc.bind_constant(M, 512)
            idxc.bind_constant(K, 512)
            idxc.bind_constant(GRID_N, 512 // BLOCK_SIZE)
            idxc.bind_constant(GRID_M, 512 // BLOCK_SIZE)
            idxc.finalize()

            sig = kernel_codegen.KernelSignature()
            sig.add_from_graph_placeholders(trace.get_root_graph())
            sig.add_grid(gemm_kernel.grid_type)
            print(sig)
            bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(
                sig, mb
            )
            emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
            try:
                emitter.emit()
            finally:
                emitter.finish()
                print(mb.module_op.get_asm())
            mb.module_op.verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

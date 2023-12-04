import unittest

import torch

from shark_turbine.kernel._support.indexing import *

M = sym.M
N = sym.N


class Test(unittest.TestCase):
    def testGridRepr(self):
        self.assertEqual("Grid", repr(Grid))
        self.assertEqual("Grid[M]", repr(Grid[M]))
        self.assertEqual("Grid[M, N]", repr(Grid[sym.M, sym.N]))

    def testGridAttrs(self):
        T = Grid[M, N]
        self.assertIs(T.symbolic_shape[0], M)
        self.assertIs(T.symbolic_shape[1], N)
        self.assertEqual(2, T.rank)

    def testGenericGridInstance(self):
        g = Grid(1, 2, 3)
        self.assertEqual(3, len(g))
        self.assertEqual(1, g[0])
        self.assertEqual([1, 2, 3], list(g))
        self.assertEqual(3, g.rank)

    def testShapedGridInstance(self):
        G = Grid[M, N]
        with self.assertRaisesRegex(ValueError, "mismatched symbolic rank"):
            g = G(1, 2, 3)

        g = G(2, 3)
        self.assertEqual(2, len(g))
        self.assertEqual(2, g[0])
        self.assertEqual([2, 3], list(g))
        self.assertEqual(2, g.rank)

    def testKernelBufferRepr(self):
        self.assertEqual("KernelBuffer", repr(KernelBuffer))
        self.assertEqual("KernelBuffer[M]", repr(KernelBuffer[sym.M]))
        self.assertEqual("KernelBuffer[M, N]", repr(KernelBuffer[sym.M, sym.N]))

    def testKernelBufferAttrs(self):
        T = KernelBuffer[M, N]
        self.assertIs(T.symbolic_shape[0], M)
        self.assertIs(T.symbolic_shape[1], N)
        self.assertEqual(2, T.rank)

    def testKernelBufferGenericInstance(self):
        kb = KernelBuffer(torch.empty((3, 4)))
        self.assertEqual(2, kb.rank)

    def testKernelBufferInstance(self):
        T1 = KernelBuffer[M]
        with self.assertRaisesRegex(ValueError, "mismatched symbolic rank"):
            T1(torch.empty((3, 4)))
        kb = T1(torch.empty((3,)))
        self.assertEqual(1, kb.rank)
        self.assertEqual((M,), kb.symbolic_shape)

    def testUsageAndElementTypeInstance(self):
        T = InputBuffer[M].of(torch.float16)
        self.assertEqual("InputBuffer[M].of(torch.float16)", repr(T))


if __name__ == "__main__":
    unittest.main()

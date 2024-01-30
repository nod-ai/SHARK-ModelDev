import logging
import unittest


from shark_turbine.kernel._support.indexing import (
    IndexingContext,
    sym,
)

from shark_turbine.kernel.compiler.analysis import (
    SliceAnalysis,
    _norm_slice_spec,
)

M = sym.M
N = sym.N
K = sym.K


class SliceAnalysisTest(unittest.TestCase):
    def testNorm(self):
        self.assertEqual([1], (_norm_slice_spec(1, 1)))
        self.assertEqual([1], (_norm_slice_spec(1, (1,))))
        self.assertEqual([slice(1, None, 1)], (_norm_slice_spec(1, (slice(1, None)))))
        self.assertEqual([None, 2], (_norm_slice_spec(2, (None, 2))))
        self.assertEqual(
            [1, 2],
            (_norm_slice_spec(2, (1, ..., 2))),
        )
        self.assertEqual(
            [1, 2],
            (_norm_slice_spec(1, (1, ..., 2))),
        )
        self.assertEqual(
            [
                None,
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                2,
            ],
            (_norm_slice_spec(5, (None, ..., 2))),
        )

    def testSymbolic(self):
        with IndexingContext() as ctx:
            sa = SliceAnalysis((M, N, K), (1, slice(None), slice(2, K), None))
            self.assertEqual(
                "[1, slice(0, N, 1), slice(2, K, 1), None]",
                repr(sa.slices),
            )
            self.assertEqual("[1, N, K - 2, None]", repr(sa.symbolic_shape))

    def testStatic(self):
        with IndexingContext() as ctx:
            ctx.bind_constant(M, 20)
            ctx.bind_constant(N, 30)
            ctx.bind_constant(K, 5)
            ctx.finalize()

            sa = SliceAnalysis((M, N, K), (1, slice(None), slice(2, K), None))
            self.assertEqual(
                "[1, slice(0, 30, 1), slice(2, 5, 1), None]",
                repr(sa.slices),
            )
            self.assertEqual("[1, 30, 3, None]", repr(sa.symbolic_shape))

    def testRejectReverseStep(self):
        with IndexingContext() as ctx:
            with self.assertRaisesRegex(IndexError, "Reverse step not allowed"):
                sa = SliceAnalysis(
                    (M, N, K),
                    (1, slice(None), slice(2, K, -K), None),
                    allow_non_unit_step=True,
                    allow_reverse_step=False,
                )

    def testRejectNonUnitStep(self):
        with IndexingContext() as ctx:
            with self.assertRaisesRegex(IndexError, "Non-unit step not allowed"):
                sa = SliceAnalysis(
                    (M, N, K),
                    (1, slice(None), slice(2, K, K), None),
                    allow_non_unit_step=False,
                    allow_reverse_step=False,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

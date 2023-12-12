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
        self.assertEqual([slice(1, 0, 0)], (_norm_slice_spec(1, 1)))
        self.assertEqual([slice(1, 0, 0)], (_norm_slice_spec(1, (1,))))
        self.assertEqual(
            [slice(1, None, None)], (_norm_slice_spec(1, (slice(1, None))))
        )
        self.assertEqual([None, slice(2, 0, 0)], (_norm_slice_spec(2, (None, 2))))
        self.assertEqual(
            [slice(1, 0, 0), slice(2, 0, 0)],
            (_norm_slice_spec(2, (1, ..., 2))),
        )
        self.assertEqual(
            [slice(1, 0, 0), slice(2, 0, 0)],
            (_norm_slice_spec(1, (1, ..., 2))),
        )
        self.assertEqual(
            [
                None,
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(2, 0, 0),
            ],
            (_norm_slice_spec(5, (None, ..., 2))),
        )

    def testSymbolic(self):
        with IndexingContext() as ctx:
            ctx.bind_constant(M, 20)
            ctx.bind_constant(N, 30)
            ctx.bind_constant(K, 5)

            sa = SliceAnalysis((M, N, K), (1, slice(None), slice(2, K), None))
            sa.normalize_symbolic_ranges()
            self.assertEqual(
                "[slice(1, 0, 0), slice(0, Symbol(N), 1), slice(2, Symbol(K), 1), None]",
                repr(sa.slices),
            )
            self.assertEqual([1, 30, 3, None], sa.symbolic_shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

import unittest
import numpy as np


from frobenius import solve


class Test(unittest.TestCase):

    def test(self):
        # f' = a f
        # or
        # d f - a x f = 0
        a = 3
        mxA = np.array([
            [ # d^0
                [[0]], # x^0
                [[-a]], # x^1
            ],
            [ # d^1
                [[1]], # x^0
                [[0]], # x^1
            ],
        ])
        s = solve(mxA, min_terms=4)
        self.assertEqual(len(s), 1)
        self.assertEqual(len(s[0]), 2)
        self.assertAlmostEqual(s[0][0], 0)
        g = s[0][1]
        self.assertEqual(len(g), 1)
        self.assertEqual(len(g[0]), 1)
        gg = g[0][0]
        np.testing.assert_allclose(gg, [[1], [a], [a**2 / 2], [a**3 / 6]])

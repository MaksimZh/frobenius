import unittest
import numpy as np


from frobenius import solve


class Test(unittest.TestCase):

    def test_1(self):
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
        lj, gj = s[0]
        self.assertAlmostEqual(lj, 0)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g, [[1], [a], [a**2 / 2], [a**3 / 6]])


    def test_2(self):
        # f'' - (a + b) f' + a b f = o
        # or
        # d^2 f + [-1 - (a + b) x] d f + a b x^2 f = 0
        a = 3
        b = 5
        mxA = np.array([
            [ # d^0
                [[0]], # x^0
                [[0]], # x^1
                [[a * b]], # x^2
            ],
            [ # d^1
                [[-1]], # x^0
                [[-(a + b)]], # x^1
                [[0]], # x^2
            ],
            [ # d^2
                [[1]], # x^0
                [[0]], # x^1
                [[0]], # x^2
            ],
        ])
        s = solve(mxA, min_terms=4)
        self.assertEqual(len(s), 2)

        lj, gj = s[0]
        self.assertAlmostEqual(lj, 0)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g, [
            [1],
            [a + b],
            [(a**2 + a * b + b**2) / 2],
            [(a**3 + a**2 * b + a * b**2 + b**3) / 6],
            [(a**4 + a**3 * b + a**2 * b**2 + a * b**3 + b**4) / 24],
            ])

        lj, gj = s[1]
        self.assertAlmostEqual(lj, 1)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g, [
            [1],
            [(a + b) / 2],
            [(a**2 + a * b + b**2) / 6],
            [(a**3 + a**2 * b + a * b**2 + b**3) / 24],
            ])


    def test_5(self):
        mxA = np.array([
            [ # d^0
                [[-72]], # x^0
            ],
            [ # d^1
                [[156]], # x^0
            ],
            [ # d^2
                [[-134]], # x^0
            ],
            [ # d^3
                [[57]], # x^0
            ],
            [ # d^4
                [[-12]], # x^0
            ],
            [ # d^5
                [[1]], # x^0
            ],
        ])
        s = solve(mxA, min_terms=4, lambda_roots=[2, 3])

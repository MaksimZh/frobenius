import unittest
import numpy as np


from frobenius import solve


class Test(unittest.TestCase):

    def test(self):
        # f' = a f
        # or
        # d f - a x f = 0
        a = 2
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
        s = solve(mxA, min_terms=3)
        self.assertEqual(len(s), 1)

import unittest
import numpy as np


from frobenius.apoly import ArrayPoly
from frobenius.smith import smith


class Test(unittest.TestCase):

    def test1(self):
        p1 = ArrayPoly(np.array([2, 1]))
        p2 = ArrayPoly(np.array([3, 1]))
        a = 5 * (p1**2 * p2)[..., np.newaxis, np.newaxis]
        x, y, kappa = smith(a, p1)
        np.testing.assert_allclose(x.coefs, [[[1]]])
        np.testing.assert_allclose(y.coefs, [[[5]]])
        np.testing.assert_allclose(kappa, [2])
        x, y, kappa = smith(a, p2)
        np.testing.assert_allclose(x.coefs, [[[1]]])
        np.testing.assert_allclose(y.coefs, [[[5]]])
        np.testing.assert_allclose(kappa, [1])

    def test2(self):
        a = ArrayPoly(np.array([
            [[4, 6], [-4, -2]],
            [[0, -6], [4, 4]],
            [[6, 7], [-2, -5]],
            [[0, -5], [2, 4]],
            [[2, 6], [0, -4]],
            [[0, -1], [0, 3]],
            [[0, 4], [0, -1]],
            [[0, 0], [0, 1]],
            [[0, 1], [0, 0]],
            ]))
        p1 = ArrayPoly(np.array([-1, 1]))
        x, y, kappa = smith(a, p1)
        y1 = ArrayPoly(y.coefs.copy())
        print(y1.coefs)
        for i in range(y1.shape[0]):
            y1[:, i] *= p1[np.newaxis]**kappa[i]
        print(y1.coefs)

Test().test2()

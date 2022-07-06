import unittest
import numpy as np


from frobenius.apoly import ArrayPoly
from frobenius.smith import smith


class Test(unittest.TestCase):

    def test(self):
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

import unittest
import numpy as np


from frobenius.apoly import ArrayPoly

coefs_3 = np.arange(1, 4) * 10

coefs_4_2x3 = \
    10 * np.arange(1, 5)[:, np.newaxis, np.newaxis] + \
    np.arange(1, 7).reshape(2, 3)


class Test(unittest.TestCase):
    
    def test_basic_3(self):
        coefs = coefs_3
        a = ArrayPoly(coefs)
        np.testing.assert_allclose(a.coefs, coefs)
        self.assertEqual(a.ndim, 0)
        self.assertEqual(a.npow, 3)
        self.assertEqual(a.shape, ())
        x = 2
        np.testing.assert_allclose(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        x = np.array([2, 3])
        np.testing.assert_allclose(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        x = np.array([[2, 3, 4], [5, 6, 7]])
        np.testing.assert_allclose(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)

    def test_basic_4_2x3(self):
        coefs = coefs_4_2x3
        a = ArrayPoly(coefs)
        np.testing.assert_allclose(a.coefs, coefs)
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.npow, 4)
        self.assertEqual(a.shape, (2, 3))
        x = 2
        np.testing.assert_allclose(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2 + coefs[3] * x**3)
        x = np.array([2, 3])
        xa = x[..., np.newaxis, np.newaxis]
        np.testing.assert_allclose(a(x),
            coefs[0, np.newaxis] + \
            coefs[1, np.newaxis] * xa + \
            coefs[2, np.newaxis] * xa**2 + \
            coefs[3, np.newaxis] * xa**3)
        x = np.array([[2, 3, 4], [5, 6, 7]])
        xa = x[..., np.newaxis, np.newaxis]
        np.testing.assert_allclose(a(x),
            coefs[0, np.newaxis, np.newaxis] + \
            coefs[1, np.newaxis, np.newaxis] * xa + \
            coefs[2, np.newaxis, np.newaxis] * xa**2 + \
            coefs[3, np.newaxis, np.newaxis] * xa**3)

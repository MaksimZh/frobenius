import unittest
import numpy as np


from frobenius.apoly import ArrayPoly


def genCoefs(npow, dims = ()):
    if not isinstance(dims, tuple):
        dims = (dims,)
    if len(dims) == 0:
        return np.arange(1, npow + 1)
    n = np.prod(dims)
    n1 = n
    f = 1
    while n1 > 0:
        f *= 10
        n1 //= 10
    s = (slice(None),) + (np.newaxis,) * len(dims)
    return f * np.arange(1, npow + 1)[s] + \
        np.arange(1, n + 1).reshape(*dims)


class TestGenCoefs(unittest.TestCase):

    def test(self):
        np.testing.assert_equal(genCoefs(3), [1, 2, 3])
        np.testing.assert_equal(genCoefs(3, 5), [
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
            [31, 32, 33, 34, 35],
            ])
        np.testing.assert_equal(genCoefs(4, (2, 3)), [
            [[11, 12, 13], [14, 15, 16]],
            [[21, 22, 23], [24, 25, 26]],
            [[31, 32, 33], [34, 35, 36]],
            [[41, 42, 43], [44, 45, 46]],
            ])
        np.testing.assert_equal(genCoefs(3, (5, 2, 4)), [
            [
                [[101, 102, 103, 104], [105, 106, 107, 108]],
                [[109, 110, 111, 112], [113, 114, 115, 116]],
                [[117, 118, 119, 120], [121, 122, 123, 124]],
                [[125, 126, 127, 128], [129, 130, 131, 132]],
                [[133, 134, 135, 136], [137, 138, 139, 140]],
            ],
            [
                [[201, 202, 203, 204], [205, 206, 207, 208]],
                [[209, 210, 211, 212], [213, 214, 215, 216]],
                [[217, 218, 219, 220], [221, 222, 223, 224]],
                [[225, 226, 227, 228], [229, 230, 231, 232]],
                [[233, 234, 235, 236], [237, 238, 239, 240]],
            ],
            [
                [[301, 302, 303, 304], [305, 306, 307, 308]],
                [[309, 310, 311, 312], [313, 314, 315, 316]],
                [[317, 318, 319, 320], [321, 322, 323, 324]],
                [[325, 326, 327, 328], [329, 330, 331, 332]],
                [[333, 334, 335, 336], [337, 338, 339, 340]],
            ],
            ])


class TestBasic(unittest.TestCase):
    
    def test_3(self):
        coefs = genCoefs(3)
        a = ArrayPoly(coefs)
        np.testing.assert_equal(a.coefs, coefs)
        self.assertEqual(a.ndim, 0)
        self.assertEqual(a.npow, 3)
        self.assertEqual(a.shape, ())
        x = 2
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        x = np.array([2, 3])
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        x = np.array([[2, 3, 4], [5, 6, 7]])
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)

    def test_3_5(self):
        coefs = genCoefs(3, 5)
        a = ArrayPoly(coefs)
        np.testing.assert_equal(a.coefs, coefs)
        self.assertEqual(a.ndim, 1)
        self.assertEqual(a.npow, 3)
        self.assertEqual(a.shape, (5,))
        x = 2
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        x = np.array([2, 3])
        xa = x[..., np.newaxis]
        np.testing.assert_equal(a(x),
            coefs[0, np.newaxis] + \
            coefs[1, np.newaxis] * xa + \
            coefs[2, np.newaxis] * xa**2)
        x = np.array([[2, 3, 4], [5, 6, 7]])
        xa = x[..., np.newaxis]
        np.testing.assert_equal(a(x),
            coefs[0, np.newaxis, np.newaxis] + \
            coefs[1, np.newaxis, np.newaxis] * xa + \
            coefs[2, np.newaxis, np.newaxis] * xa**2)

    def test_4_2x3(self):
        coefs = genCoefs(4, (2, 3))
        a = ArrayPoly(coefs)
        np.testing.assert_equal(a.coefs, coefs)
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.npow, 4)
        self.assertEqual(a.shape, (2, 3))
        x = 2
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2 + coefs[3] * x**3)
        x = np.array([2, 3])
        xa = x[..., np.newaxis, np.newaxis]
        np.testing.assert_equal(a(x),
            coefs[0, np.newaxis] + \
            coefs[1, np.newaxis] * xa + \
            coefs[2, np.newaxis] * xa**2 + \
            coefs[3, np.newaxis] * xa**3)
        x = np.array([[2, 3, 4], [5, 6, 7]])
        xa = x[..., np.newaxis, np.newaxis]
        np.testing.assert_equal(a(x),
            coefs[0, np.newaxis, np.newaxis] + \
            coefs[1, np.newaxis, np.newaxis] * xa + \
            coefs[2, np.newaxis, np.newaxis] * xa**2 + \
            coefs[3, np.newaxis, np.newaxis] * xa**3)

    def test_3_5x2x4(self):
        coefs = genCoefs(3, (5, 2, 4))
        a = ArrayPoly(coefs)
        np.testing.assert_equal(a.coefs, coefs)
        self.assertEqual(a.ndim, 3)
        self.assertEqual(a.npow, 3)
        self.assertEqual(a.shape, (5, 2, 4))
        x = 2
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        x = np.array([2, 3])
        xa = x[..., np.newaxis, np.newaxis, np.newaxis]
        np.testing.assert_equal(a(x),
            coefs[0, np.newaxis] + \
            coefs[1, np.newaxis] * xa + \
            coefs[2, np.newaxis] * xa**2)
        x = np.array([[2, 3, 4], [5, 6, 7]])
        xa = x[..., np.newaxis, np.newaxis, np.newaxis]
        np.testing.assert_equal(a(x),
            coefs[0, np.newaxis, np.newaxis] + \
            coefs[1, np.newaxis, np.newaxis] * xa + \
            coefs[2, np.newaxis, np.newaxis] * xa**2)


class TestIndex(unittest.TestCase):

    def test_3_5(self):
        ca = genCoefs(3, 5)
        a = ArrayPoly(ca)
        x = 5
        np.testing.assert_equal(a[...](x), a(x)[...])
        np.testing.assert_equal(a[:](x), a(x)[:])
        np.testing.assert_equal(a[2](x), a(x)[2])
        np.testing.assert_equal(a[1:4](x), a(x)[1:4])
        cb = genCoefs(3)
        cc = ca.copy()
        cc[:, 1] = cb
        a[1] = ArrayPoly(cb)
        np.testing.assert_equal(a.coefs, cc)

    def test_3_5x6(self):
        a = ArrayPoly(genCoefs(3, (5, 6)))
        x = 5
        np.testing.assert_equal(a[...](x), a(x)[...])
        np.testing.assert_equal(a[:](x), a(x)[:])
        np.testing.assert_equal(a[2](x), a(x)[2])
        np.testing.assert_equal(a[1:4](x), a(x)[1:4])
        np.testing.assert_equal(a[:, :](x), a(x)[:, :])
        np.testing.assert_equal(a[2, 3](x), a(x)[2, 3])
        np.testing.assert_equal(a[:, 3](x), a(x)[:, 3])
        np.testing.assert_equal(a[2, 1:4](x), a(x)[2, 1:4])
        np.testing.assert_equal(a[2:5, 1:4](x), a(x)[2:5, 1:4])

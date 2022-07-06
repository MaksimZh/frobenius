import unittest
import numpy as np


from frobenius.apoly import ArrayPoly


class TestBasic(unittest.TestCase):

    def test_0d(self):
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

    def test_1d(self):
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

    def test_2d(self):
        coefs = genCoefs(4, 2, 3)
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

    def test_3d(self):
        coefs = genCoefs(3, 5, 2, 4)
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

    def test_1d(self):

        ca = genCoefs(3, 5)

        a = ArrayPoly(ca)
        np.testing.assert_equal(a[...].coefs, a.coefs[:, ...])
        np.testing.assert_equal(a[:].coefs, a.coefs[:, :])
        np.testing.assert_equal(a[2].coefs, a.coefs[:, 2])
        np.testing.assert_equal(a[1:4].coefs, a.coefs[:, 1:4])

        a = ArrayPoly(ca)
        cb = genCoefs(3, 2)
        cc = ca.copy()
        cc[:, 1:3] = cb
        a[1:3] = ArrayPoly(cb)
        np.testing.assert_equal(a.coefs, cc)

        a = ArrayPoly(ca)
        cb = genCoefs(2, 2)
        cc = ca.copy()
        cc[:2, 1:3] = cb
        a[1:3] = ArrayPoly(cb)
        np.testing.assert_equal(a.coefs, cc)

        a = ArrayPoly(ca)
        cb = genCoefs(6, 2)
        cc = np.zeros((6, 5))
        cc[:3] = ca
        cc[:, 1:3] = cb
        a[1:3] = ArrayPoly(cb)
        np.testing.assert_equal(a.coefs, cc)


    def test_2d(self):

        ca = genCoefs(3, 5, 6)

        a = ArrayPoly(ca)
        np.testing.assert_equal(a[...].coefs, a.coefs[:, ...])
        np.testing.assert_equal(a[:].coefs, a.coefs[:, :])
        np.testing.assert_equal(a[2].coefs, a.coefs[:, 2])
        np.testing.assert_equal(a[1:4].coefs, a.coefs[:, 1:4])
        np.testing.assert_equal(a[:, :].coefs, a.coefs[:, :, :])
        np.testing.assert_equal(a[2, 3].coefs, a.coefs[:, 2, 3])
        np.testing.assert_equal(a[:, 3].coefs, a.coefs[:, :, 3])
        np.testing.assert_equal(a[2, 1:4].coefs, a.coefs[:, 2, 1:4])
        np.testing.assert_equal(a[2:4, 1:5].coefs, a.coefs[:, 2:4, 1:5])

        a = ArrayPoly(ca)
        cb = genCoefs(3, 2, 4)
        cc = ca.copy()
        cc[:, 2:4, 1:5] = cb
        a[2:4, 1:5] = ArrayPoly(cb)
        np.testing.assert_equal(a.coefs, cc)

        a = ArrayPoly(ca)
        cb = genCoefs(2, 2, 4)
        cc = ca.copy()
        cc[:2, 2:4, 1:5] = cb
        a[2:4, 1:5] = ArrayPoly(cb)
        np.testing.assert_equal(a.coefs, cc)

        a = ArrayPoly(ca)
        cb = genCoefs(6, 2, 4)
        cc = np.zeros((6, 5, 6))
        cc[:3] = ca
        cc[:, 2:4, 1:5] = cb
        a[2:4, 1:5] = ArrayPoly(cb)
        np.testing.assert_equal(a.coefs, cc)


class TestArithmetic(unittest.TestCase):

    def test_pos(self):
        a = ArrayPoly(genCoefs(4, 2, 3))
        np.testing.assert_equal((+a).coefs, a.coefs)
    
    def test_neg(self):
        a = ArrayPoly(genCoefs(4, 2, 3))
        np.testing.assert_equal((-a).coefs, -a.coefs)

    def test_mul_scalar(self):
        a = ArrayPoly(genCoefs(4, 2, 3))
        np.testing.assert_equal((a * 5).coefs, a.coefs * 5)
        np.testing.assert_equal((5 * a).coefs, 5 * a.coefs)
        np.testing.assert_allclose((a / 5).coefs, a.coefs / 5)
        np.testing.assert_allclose((a // 3).coefs, a.coefs // 3)

    def test_add(self):
        x = np.arange(4)
        a = ArrayPoly(genCoefs(4, 2, 3))
        b = ArrayPoly(genCoefs(4, 2, 3))
        np.testing.assert_equal((a + b)(x), a(x) + b(x))
        b = ArrayPoly(genCoefs(2, 2, 3))
        np.testing.assert_equal((a + b)(x), a(x) + b(x))
        b = ArrayPoly(genCoefs(6, 2, 3))
        np.testing.assert_equal((a + b)(x), a(x) + b(x))
        b = ArrayPoly(genCoefs(4, 2, 1))
        np.testing.assert_equal((a + b)(x), a(x) + b(x))
        a = ArrayPoly(genCoefs(3, 5, 2, 1))
        b = ArrayPoly(genCoefs(2, 5, 2, 3))
        np.testing.assert_equal((a + b)(x), a(x) + b(x))

    def test_sub(self):
        x = np.arange(4)
        a = ArrayPoly(genCoefs(4, 2, 3))
        b = ArrayPoly(genCoefs(4, 2, 3))
        np.testing.assert_equal((a - b)(x), a(x) - b(x))
        b = ArrayPoly(genCoefs(2, 2, 3))
        np.testing.assert_equal((a - b)(x), a(x) - b(x))
        b = ArrayPoly(genCoefs(6, 2, 3))
        np.testing.assert_equal((a - b)(x), a(x) - b(x))
        b = ArrayPoly(genCoefs(4, 2, 1))
        np.testing.assert_equal((a - b)(x), a(x) - b(x))
        a = ArrayPoly(genCoefs(3, 5, 2, 1))
        b = ArrayPoly(genCoefs(2, 5, 2, 3))
        np.testing.assert_equal((a - b)(x), a(x) - b(x))

    def test_mul(self):
        x = np.arange(4)
        a = ArrayPoly(genCoefs(4, 2, 3))
        b = ArrayPoly(genCoefs(2, 2, 3))
        np.testing.assert_equal((a * b)(x), a(x) * b(x))
        b = ArrayPoly(genCoefs(2, 2, 1))
        np.testing.assert_equal((a * b)(x), a(x) * b(x))
        a = ArrayPoly(genCoefs(3, 5, 2, 1))
        b = ArrayPoly(genCoefs(2, 5, 2, 3))
        np.testing.assert_equal((a * b)(x), a(x) * b(x))

    def test_matmul(self):
        x = np.arange(4)
        a = ArrayPoly(genCoefs(4, 5, 2, 3))
        b = ArrayPoly(genCoefs(2, 5, 3, 4))
        np.testing.assert_equal((a @ b)(x), a(x) @ b(x))
        b = ArrayPoly(genCoefs(2, 1, 3, 4))
        np.testing.assert_equal((a @ b)(x), a(x) @ b(x))

    def test_pow(self):
        x = np.arange(4)
        a = ArrayPoly(genCoefs(4, 2, 3))
        np.testing.assert_equal((a ** 2)(x), a(x) ** 2)


class TestSubst(unittest.TestCase):

    def test_0d_0d(self):
        a = ArrayPoly(genCoefs(3))
        b = ArrayPoly(genCoefs(2))
        x = np.arange(4)
        np.testing.assert_equal(a(b)(x), a(b(x)))

    def test_1d_0d(self):
        a = ArrayPoly(genCoefs(3, 5))
        b = ArrayPoly(genCoefs(2))
        x = np.arange(4)
        np.testing.assert_equal(a(b)(x), a(b(x)))

    def test_1d_1d(self):
        a = ArrayPoly(genCoefs(3, 4))
        b = ArrayPoly(genCoefs(2, 4))
        x = np.arange(4)
        bx = b(x)
        np.testing.assert_equal(a(b)(x),
            a.coefs[0] + a.coefs[1] * bx + a.coefs[2] * bx ** 2)

    def test_2d_3d(self):
        a = ArrayPoly(genCoefs(3, 5, 2, 1))
        b = ArrayPoly(genCoefs(2, 5, 2, 3))
        x = np.arange(4)
        bx = b(x)
        np.testing.assert_equal(a(b)(x),
            a.coefs[0] + a.coefs[1] * bx + a.coefs[2] * bx ** 2)


from frobenius.apoly import _it

class TestIT(unittest.TestCase):
    
    def test(self):
        self.assertTupleEqual(_it(()), (slice(None),))
        self.assertTupleEqual(_it(-1), (Ellipsis,))
        self.assertTupleEqual(_it(0), ())
        self.assertTupleEqual(_it(1), (np.newaxis,))
        self.assertTupleEqual(_it(2), (np.newaxis, np.newaxis))
        self.assertTupleEqual(_it(2, (), -1, (), 1),
            (np.newaxis, np.newaxis, slice(None), Ellipsis, slice(None), np.newaxis))


def genCoefs(*args):
    if len(args) < 2:
        return np.arange(1, args[0] + 1)
    n = np.prod(args[1:])
    n1 = n
    f = 1
    while n1 > 0:
        f *= 10
        n1 //= 10
    s = (slice(None),) + (np.newaxis,) * (len(args) - 1)
    return f * np.arange(1, args[0] + 1)[s] + \
        np.arange(1, n + 1).reshape(*args[1:])


class TestGenCoefs(unittest.TestCase):

    def test(self):
        np.testing.assert_equal(genCoefs(3), [1, 2, 3])
        np.testing.assert_equal(genCoefs(3, 5), [
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
            [31, 32, 33, 34, 35],
            ])
        np.testing.assert_equal(genCoefs(4, 2, 3), [
            [[11, 12, 13], [14, 15, 16]],
            [[21, 22, 23], [24, 25, 26]],
            [[31, 32, 33], [34, 35, 36]],
            [[41, 42, 43], [44, 45, 46]],
            ])
        np.testing.assert_equal(genCoefs(3, 5, 2, 4), [
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

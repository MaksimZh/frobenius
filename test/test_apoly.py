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
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            coefs[1] + 2 * coefs[2] * x)
        np.testing.assert_equal(a(x, deriv=2), 2 * coefs[2])
        np.testing.assert_equal(a(x, deriv=3), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=4), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=5), 0 * coefs[0])
        
        x = np.array([2, 3])
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            coefs[1] + 2 * coefs[2] * x)
        np.testing.assert_equal(a(x, deriv=2), 2 * coefs[2])
        np.testing.assert_equal(a(x, deriv=3), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=4), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=5), 0 * coefs[0])
        
        x = np.array([[2, 3, 4], [5, 6, 7]])
        np.testing.assert_equal(a(x),
            coefs[0] + coefs[1] * x + coefs[2] * x**2)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            coefs[1] + 2 * coefs[2] * x)
        np.testing.assert_equal(a(x, deriv=2), 2 * coefs[2])
        np.testing.assert_equal(a(x, deriv=3), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=4), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=5), 0 * coefs[0])


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
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            coefs[1] + 2 * coefs[2] * x)
        np.testing.assert_equal(a(x, deriv=2), 2 * coefs[2])
        np.testing.assert_equal(a(x, deriv=3), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=4), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=5), 0 * coefs[0])
        
        x = np.array([2, 3])
        xa = x[..., np.newaxis]
        ca = coefs[:, np.newaxis] * np.ones_like(xa)
        np.testing.assert_equal(a(x),
            ca[0] + ca[1] * xa + ca[2] * xa**2)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            ca[1] + ca[2] * xa * 2)
        np.testing.assert_equal(a(x, deriv=2), ca[2] * 2)
        np.testing.assert_equal(a(x, deriv=3), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=4), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=5), ca[0] * 0)
        
        x = np.array([[2, 3, 4], [5, 6, 7]])
        xa = x[..., np.newaxis]
        ca = coefs[:, np.newaxis, np.newaxis] * np.ones_like(xa)
        np.testing.assert_equal(a(x),
            ca[0] + ca[1] * xa + ca[2] * xa**2)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            ca[1] + ca[2] * xa * 2)
        np.testing.assert_equal(a(x, deriv=2), ca[2] * 2)
        np.testing.assert_equal(a(x, deriv=3), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=4), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=5), ca[0] * 0)


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
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            coefs[1] + 2 * coefs[2] * x + 3 * coefs[3] * x**2)
        np.testing.assert_equal(a(x, deriv=2),
            2 * coefs[2] + 6 * coefs[3] * x)
        np.testing.assert_equal(a(x, deriv=3), 6 * coefs[3])
        np.testing.assert_equal(a(x, deriv=4), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=5), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=6), 0 * coefs[0])
        
        x = np.array([2, 3])
        xa = x[..., np.newaxis, np.newaxis]
        ca = coefs[:, np.newaxis] * np.ones_like(xa)
        np.testing.assert_equal(a(x),
            ca[0] + ca[1] * xa + ca[2] * xa**2 + ca[3] * xa**3)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            ca[1] + ca[2] * xa * 2 + ca[3] * xa**2 * 3)
        np.testing.assert_equal(a(x, deriv=2),
            ca[2] * 2 + ca[3] * xa * 6)
        np.testing.assert_equal(a(x, deriv=3), ca[3] * 6)
        np.testing.assert_equal(a(x, deriv=4), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=5), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=6), ca[0] * 0)
        
        x = np.array([[2, 3, 4], [5, 6, 7]])
        xa = x[..., np.newaxis, np.newaxis]
        ca = coefs[:, np.newaxis, np.newaxis] * np.ones_like(xa)
        np.testing.assert_equal(a(x),
            ca[0] + ca[1] * xa + ca[2] * xa**2 + ca[3] * xa**3)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            ca[1] + ca[2] * xa * 2 + ca[3] * xa**2 * 3)
        np.testing.assert_equal(a(x, deriv=2),
            ca[2] * 2 + ca[3] * xa * 6)
        np.testing.assert_equal(a(x, deriv=3), ca[3] * 6)
        np.testing.assert_equal(a(x, deriv=4), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=5), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=6), ca[0] * 0)


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
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            coefs[1] + 2 * coefs[2] * x)
        np.testing.assert_equal(a(x, deriv=2), 2 * coefs[2])
        np.testing.assert_equal(a(x, deriv=3), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=4), 0 * coefs[0])
        np.testing.assert_equal(a(x, deriv=5), 0 * coefs[0])
         
        x = np.array([2, 3])
        xa = x[..., np.newaxis, np.newaxis, np.newaxis]
        ca = coefs[:, np.newaxis] * np.ones_like(xa)
        np.testing.assert_equal(a(x),
            ca[0] + ca[1] * xa + ca[2] * xa**2)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            ca[1] + ca[2] * xa * 2)
        np.testing.assert_equal(a(x, deriv=2), ca[2] * 2)
        np.testing.assert_equal(a(x, deriv=3), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=4), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=5), ca[0] * 0)
          
        x = np.array([[2, 3, 4], [5, 6, 7]])
        xa = x[..., np.newaxis, np.newaxis, np.newaxis]
        ca = coefs[:, np.newaxis, np.newaxis] * np.ones_like(xa)
        np.testing.assert_equal(a(x),
            ca[0] + ca[1] * xa + ca[2] * xa**2)
        np.testing.assert_equal(a(x, deriv=0), a(x))
        np.testing.assert_equal(a(x, deriv=1),
            ca[1] + ca[2] * xa * 2)
        np.testing.assert_equal(a(x, deriv=2), ca[2] * 2)
        np.testing.assert_equal(a(x, deriv=3), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=4), ca[0] * 0)
        np.testing.assert_equal(a(x, deriv=5), ca[0] * 0)
  

class TestCoefs(unittest.TestCase):

    def test_list(self):
        for c in [
            [1, 2, 3],
            [[1, 2, 3], [4, 5, 6]],
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        ]:
            a = ArrayPoly(c)
            self.assertIsInstance(a.coefs, np.ndarray)
            np.testing.assert_equal(a.coefs, c)

    def test_scalar(self):
        a = ArrayPoly(42)
        self.assertEqual(a.coefs.shape, (1,))
        np.testing.assert_equal(a.coefs, [42])


class TestIndex(unittest.TestCase):

    def test_1d(self):

        ca = genCoefs(3, 5)

        a = ArrayPoly(ca)
        np.testing.assert_equal(a[np.newaxis].coefs, a.coefs[:, np.newaxis])
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
        cc[2:, 1:3] = 0
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
        np.testing.assert_equal(a[np.newaxis].coefs, a.coefs[:, np.newaxis])
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
        cc[2:, 2:4, 1:5] = 0
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


class TestDivMod(unittest.TestCase):

    def test_0d(self):
        a = ArrayPoly(genCoefs(5))
        b = ArrayPoly(genCoefs(3))
        q, r = divmod(a, b)
        x = np.arange(4)
        np.testing.assert_allclose(a(x), (b * q + r)(x))
        np.testing.assert_allclose((a // b)(x), q(x))
        np.testing.assert_allclose((a % b)(x), r(x))

    def test_3d(self):
        a = ArrayPoly(genCoefs(5, 2, 3, 4))
        b = ArrayPoly(genCoefs(3))
        q, r = divmod(a, b)
        x = np.arange(4)
        np.testing.assert_allclose(a(x), (b * q + r)(x))
        np.testing.assert_allclose((a // b)(x), q(x))
        np.testing.assert_allclose((a % b)(x), r(x))


from frobenius.apoly import trim

class TestTrim(unittest.TestCase):

    def test_none(self):
        a = ArrayPoly([1, 2, 3])
        np.testing.assert_allclose(trim(a).coefs, a.coefs)
        np.testing.assert_allclose(trim(a, 1).coefs, a.coefs)
        a = ArrayPoly([[1, 2], [0, 4], [5, 0]])
        np.testing.assert_allclose(trim(a).coefs, a.coefs)
        np.testing.assert_allclose(trim(a, 1).coefs, a.coefs)

    def test_some(self):
        a = ArrayPoly([1, 2, 3, 0.5, 0, 0])
        np.testing.assert_allclose(trim(a).coefs, a.coefs[:4])
        np.testing.assert_allclose(trim(a, 1).coefs, a.coefs[:3])
        a = ArrayPoly([[1, 2], [0, 4], [0.5, 0], [0, 0]])
        np.testing.assert_allclose(trim(a).coefs, a.coefs[:3])
        np.testing.assert_allclose(trim(a, 1).coefs, a.coefs[:2])

    def test_all(self):
        a = ArrayPoly([0.5, 0, 0])
        np.testing.assert_allclose(trim(a).coefs, a.coefs[:1])
        np.testing.assert_allclose(trim(a, 1).coefs, a.coefs[:1])
        a = ArrayPoly([[0.5, 0], [0, 0]])
        np.testing.assert_allclose(trim(a).coefs, a.coefs[:1])
        np.testing.assert_allclose(trim(a, 1).coefs, a.coefs[:1])


from frobenius.apoly import det

class TestDet(unittest.TestCase):

    def test_1_1(self):
        x = np.arange(4)
        a = ArrayPoly(genCoefs(1, 1, 1))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))
        a = ArrayPoly(genCoefs(1, 5, 1, 1))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))

    def test_1_3(self):
        x = np.arange(4)
        a = ArrayPoly(genCoefs(3, 1, 1))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))
        a = ArrayPoly(genCoefs(3, 5, 1, 1))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))

    def test_4_1(self):
        x = np.arange(4)
        c = np.arange(4 * 4).reshape(1, 4, 4) + 1
        a = ArrayPoly(c + 2 * c.transpose(0, 2, 1))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))
        c = (np.arange(5 * 4 * 4).reshape(1, 5, 4, 4) + 1) % 7
        a = ArrayPoly(c + 2 * c.transpose(0, 1, 3, 2))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))

    def test_4_3(self):
        x = np.arange(4)
        c = (np.arange(3 * 4 * 4).reshape(3, 4, 4) + 1) % 7
        a = ArrayPoly(c + 2 * c.transpose(0, 2, 1))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))
        c = (np.arange(3 * 5 * 4 * 4).reshape(3, 5, 4, 4) + 1) % 7
        a = ArrayPoly(c + 2 * c.transpose(0, 1, 3, 2))
        np.testing.assert_allclose(det(a)(x), np.linalg.det(a(x)))


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

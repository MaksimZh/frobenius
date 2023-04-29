import unittest
import numpy as np


from frobenius.apoly import ArrayPoly, trim
from frobenius.smith import smith


class Test(unittest.TestCase):

    def check(self, a, p, x, y, kappa):
        y1 = ArrayPoly(y.coefs.copy())
        for i in range(y1.shape[0]):
            y1[:, i] *= p[np.newaxis]**kappa[i]
        self.assertAlmostEqual(np.sum(np.abs((a @ x - y1).coefs)), 0)

    def test1(self):
        p1 = ArrayPoly([2, 1])
        p2 = ArrayPoly([3, 1])
        a = 5 * (p1**2 * p2)[..., np.newaxis, np.newaxis]
        x, y, kappa = smith(a, p1)
        self.check(a, p1, x, y, kappa)
        x, y, kappa = smith(a, p2)
        self.check(a, p2, x, y, kappa)

    def test_Yu(self):
        a = ArrayPoly([
            [[4, 6], [-4, -2]],
            [[0, -6], [4, 4]],
            [[6, 7], [-2, -5]],
            [[0, -5], [2, 4]],
            [[2, 6], [0, -4]],
            [[0, -1], [0, 3]],
            [[0, 4], [0, -1]],
            [[0, 0], [0, 1]],
            [[0, 1], [0, 0]],
            ])
        p = ArrayPoly([-1, 1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)
        p = ArrayPoly([2, 0, 1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)
        p = ArrayPoly([1j*np.sqrt(2), 1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)
        p = ArrayPoly([-1j*np.sqrt(2), 1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)

    def test_Barkatou(self):
        a = ArrayPoly([
            [
                [0, 2, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [-6, -1, 0],
                [-4, 0, 0],
                [-1, 0, 0],
            ],
            [
                [3, 0, 0],
                [4, 0, 0],
                [0, 0, 1],
            ],
            [
                [0, 0, 0],
                [-1, 0, 0],
                [0, 0, 0],
            ]])
        p = ArrayPoly([0, 1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)
        p = ArrayPoly([2, -1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)

    def test2(self):
        a = ArrayPoly(np.array([-72, 156, -134, 57, -12, 1]).reshape(-1, 1, 1))
        p = ArrayPoly([-2, 1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)
        p = ArrayPoly([-3, 1])
        x, y, kappa = smith(a, p)
        self.check(a, p, x, y, kappa)


from frobenius.smith import ColumnCollector, ColumnQueue, ExtendibleMatrix

class Test_ColumnCollector(unittest.TestCase):

    def test_init(self):
        a = ColumnCollector(2)
        self.assertTrue(a.is_empty())

    def test_push(self):
        a = ColumnCollector(2)
        self.assertTrue(a.is_status("push_right", "NIL"))
        a.push_right(ArrayPoly([[1, 2]]))
        self.assertTrue(a.is_status("push_right", "NOT_COLUMN"))
        a.push_right(ArrayPoly([[[1], [2], [3]]]))
        self.assertTrue(a.is_status("push_right", "SIZE_MISMATCH"))
        a.push_right(ArrayPoly([[[1], [2]]]))
        self.assertTrue(a.is_status("push_right", "OK"))
        self.assertFalse(a.is_empty())

class Test_ColumnQueue(unittest.TestCase):
    
    def test_pop(self):
        a = ColumnQueue(2)
        self.assertTrue(a.is_status("pop_left", "NIL"))
        a.pop_left()
        self.assertTrue(a.is_status("pop_left", "EMPTY"))
        a.push_right(ArrayPoly([[[1], [2]]]))
        a.pop_left()
        self.assertTrue(a.is_status("pop_left", "OK"))
        self.assertTrue(a.is_empty())
        a.push_right(ArrayPoly([[[1], [2]]]))
        a.push_right(ArrayPoly([[[1], [2]]]))
        a.push_right(ArrayPoly([[[1], [2]]]))
        a.pop_left()
        self.assertTrue(a.is_status("pop_left", "OK"))
        a.pop_left()
        self.assertTrue(a.is_status("pop_left", "OK"))
        a.pop_left()
        self.assertTrue(a.is_status("pop_left", "OK"))
        a.pop_left()
        self.assertTrue(a.is_status("pop_left", "EMPTY"))
    
    def test_get(self):
        p1 = ArrayPoly([[[1], [2]], [[3], [4]]])
        p2 = ArrayPoly([[[5], [6]]])
        p3 = ArrayPoly([[[7], [8]], [[9], [10]], [[11], [12]]])
        a = ColumnQueue(2)
        self.assertTrue(a.is_status("get_left", "NIL"))
        c = a.get_left()
        self.assertTrue(a.is_status("get_left", "EMPTY"))
        a.push_right(p1)
        a.push_right(p2)
        a.push_right(p3)
        c = a.get_left()
        self.assertTrue(a.is_status("get_left", "OK"))
        np.testing.assert_almost_equal(c.coefs, p1.coefs)
        a.pop_left()
        c = a.get_left()
        self.assertTrue(a.is_status("get_left", "OK"))
        np.testing.assert_almost_equal(c.coefs, p2.coefs)
        a.pop_left()
        c = a.get_left()
        self.assertTrue(a.is_status("get_left", "OK"))
        np.testing.assert_almost_equal(c.coefs, p3.coefs)
        a.pop_left()
        c = a.get_left()
        self.assertTrue(a.is_status("get_left", "EMPTY"))

    def test_rotate_and_clear(self):
        p = [
            ArrayPoly([[[1], [2]], [[3], [4]]]),
            ArrayPoly([[[5], [6]]]),
            ArrayPoly([[[7], [8]], [[9], [10]], [[11], [12]]]),
        ]
        a = ColumnQueue(2)
        a.push_right(p[0])
        a.push_right(p[1])
        a.push_right(p[2])
        np.testing.assert_array_almost_equal(a.get_left().coefs, p[0].coefs)
        a.pop_left()
        a.push_right(p[0] * 2)
        np.testing.assert_array_almost_equal(a.get_left().coefs, p[1].coefs)
        a.pop_left()
        a.push_right(p[1] * 2)
        np.testing.assert_array_almost_equal(a.get_left().coefs, p[2].coefs)
        a.pop_left()
        a.push_right(p[2] * 2)
        np.testing.assert_array_almost_equal(a.get_left().coefs, p[0].coefs * 2)
        a.pop_left()
        np.testing.assert_array_almost_equal(a.get_left().coefs, p[1].coefs * 2)
        a.pop_left()
        np.testing.assert_array_almost_equal(a.get_left().coefs, p[2].coefs * 2)
        a.pop_left()
        self.assertTrue(a.is_empty())


class Test_ExtendibleMatrix(unittest.TestCase):

    def test_get_matrix(self):
        p = [
            ArrayPoly([[[1], [2]], [[3], [4]]]),
            ArrayPoly([[[5], [6]]]),
            ArrayPoly([[[7], [8]], [[9], [10]], [[11], [12]]]),
        ]
        a = ExtendibleMatrix(2)
        self.assertTrue(a.is_status("get_matrix", "NIL"))
        m = a.get_matrix()
        self.assertTrue(a.is_status("get_matrix", "EMPTY"))
        a.push_right(p[0])
        a.push_right(p[1])
        a.push_right(p[2])
        m = a.get_matrix()
        self.assertTrue(a.is_status("get_matrix", "OK"))
        np.testing.assert_almost_equal(
            m.coefs,
            [
                [
                    [1, 5, 7],
                    [2, 6, 8],
                ],
                [
                    [3, 0, 9],
                    [4, 0, 10],
                ],
                [
                    [0, 0, 11],
                    [0, 0, 12],
                ],
            ])

    def test_expand_column(self):
        p = [
            ArrayPoly([[[1], [2]], [[3], [4]]]),
            ArrayPoly([[[5], [6]]]),
            ArrayPoly([[[7], [8]], [[9], [10]], [[11], [12]]]),
        ]
        a = ExtendibleMatrix(2)
        self.assertTrue(a.is_status("expand_column", "NIL"))
        c = a.expand_column(p[0])
        self.assertTrue(a.is_status("expand_column", "EMPTY"))
        a.push_right(p[0])
        a.push_right(p[1])
        c = a.expand_column(ArrayPoly([[1]]))
        self.assertTrue(a.is_status("expand_column", "INVALID_COLUMN"))
        c = a.expand_column(p[0])
        self.assertTrue(a.is_status("expand_column", "OK"))
        np.testing.assert_almost_equal(c, [1, 0])
        c = a.expand_column(p[1])
        self.assertTrue(a.is_status("expand_column", "OK"))
        np.testing.assert_almost_equal(c, [0, 1])
        c = a.expand_column(p[2])
        self.assertTrue(a.is_status("expand_column", "NOT_COMPLANAR"))
        c = a.expand_column(2 * p[0] + 3 * p[1])
        self.assertTrue(a.is_status("expand_column", "OK"))
        np.testing.assert_almost_equal(c, [2, 3])

    def test_combine_columns(self):
        p = [
            ArrayPoly([[[1], [2]], [[3], [4]]]),
            ArrayPoly([[[5], [6]]]),
            ArrayPoly([[[7], [8]], [[9], [10]], [[11], [12]]]),
        ]
        a = ExtendibleMatrix(2)
        self.assertTrue(a.is_status("combine_columns", "NIL"))
        c = a.combine_columns([1])
        self.assertTrue(a.is_status("combine_columns", "EMPTY"))
        a.push_right(p[0])
        a.push_right(p[1])
        a.push_right(p[2])
        c = a.combine_columns([1])
        self.assertTrue(a.is_status("combine_columns", "SIZE_MISMATCH"))
        f = [2, ArrayPoly([3, 4]), 5]
        c = a.combine_columns(f)
        self.assertTrue(a.is_status("combine_columns", "OK"))
        np.testing.assert_almost_equal(
            c.coefs,
            (p[0] * f[0] + p[1] * f[1] + p[2] * f[2]).coefs)

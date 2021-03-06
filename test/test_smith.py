import unittest
import numpy as np


from frobenius.apoly import ArrayPoly
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

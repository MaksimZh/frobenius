import unittest
import numpy as np


from frobenius import solve


class TestSingle(unittest.TestCase):

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
        np.testing.assert_allclose(g, [
            [[[1]]],
            [[[a]]],
            [[[a**2 / 2]]],
            [[[a**3 / 6]]],
        ])


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
            [[[1]]],
            [[[a + b]]],
            [[[(a**2 + a * b + b**2) / 2]]],
            [[[(a**3 + a**2 * b + a * b**2 + b**3) / 6]]],
            [[[(a**4 + a**3 * b + a**2 * b**2 + a * b**3 + b**4) / 24]]],
        ])

        lj, gj = s[1]
        self.assertAlmostEqual(lj, 1)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g, [
            [[[1]]],
            [[[(a + b) / 2]]],
            [[[(a**2 + a * b + b**2) / 6]]],
            [[[(a**3 + a**2 * b + a * b**2 + b**3) / 24]]],
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
        # Don't remove lambda_roots parameter
        # automatic root calculation is not precise enough for this test
        s = solve(mxA, min_terms=4, lambda_roots=[2, 3])
        self.assertEqual(len(s), 2)

        lj, gj = s[0]
        self.assertAlmostEqual(lj, 2)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 3)
        
        # f = x^2
        g = gj[0][0]
        np.testing.assert_allclose(g / max(g[0]),[
            [[[1]]],
            [[[0]]],
            [[[0]]],
            [[[0]]],
            [[[0]]],
        ])
        
        # f = x^2 ln(x)
        g = gj[0][1]
        np.testing.assert_allclose(g / max(g[0]),[
            [[[0]], [[1]]],
            [[[0]], [[0]]],
            [[[0]], [[0]]],
            [[[0]], [[0]]],
            [[[0]], [[0]]],
        ])

        # f = x^2 ln^2(x)
        g = gj[0][2]
        np.testing.assert_allclose(g / max(g[0]),[
            [[[0]], [[0]], [[1]]],
            [[[0]], [[0]], [[0]]],
            [[[0]], [[0]], [[0]]],
            [[[0]], [[0]], [[0]]],
            [[[0]], [[0]], [[0]]],
        ])

        lj, gj = s[1]
        self.assertAlmostEqual(lj, 3)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 2)
        
        # f = x^3
        g = gj[0][0]
        np.testing.assert_allclose(g / max(g[0]),[
            [[[1]]],
            [[[0]]],
            [[[0]]],
            [[[0]]],
        ])
        
        # f = x^3 ln(x)
        g = gj[0][1]
        np.testing.assert_allclose(g / max(g[0]),[
            [[[0]], [[1]]],
            [[[0]], [[0]]],
            [[[0]], [[0]]],
            [[[0]], [[0]]],
        ])


    def test_bessel(self):
        n = 2
        ode_coefs_theta = np.array([
            [ # theta^0
                [[-n**2]], # x^0
                [[0]], # x^1
                [[1]], # x^2
            ],
            [ # theta^1
                [[0]], # x^0
                [[0]], # x^1
                [[0]], # x^2
            ],
            [ # theta^2
                [[1]], # x^0
                [[0]], # x^1
                [[0]], # x^2
            ],
        ])
        s = solve(ode_coefs_theta, min_terms=5)
        self.assertEqual(len(s), 2)

        lj, gj = s[0]
        self.assertAlmostEqual(lj, -2)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g / g[0, 0, 0, 0], [
            [[[1]], [[0]]],
            [[[0]], [[0]]],
            [[[1/4]], [[0]]],
            [[[0]], [[0]]],
            [[[1/64]], [[-1/16]]],
            [[[0]], [[0]]],
            [[[-11/2304]], [[1/192]]],
            [[[0]], [[0]]],
            [[[31/147456]], [[-1/6144]]],
        ])

        lj, gj = s[1]
        self.assertAlmostEqual(lj, 2)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g / g[0, 0, 0, 0], [
            [[[1]]],
            [[[0]]],
            [[[-1/12]]],
            [[[0]]],
            [[[1/384]]],
        ])



class TestSystem(unittest.TestCase):

    def test_bessel(self):
        # nth Bessel equation converted to a system of two 1st order ODEs
        # f = (y, x * y')
        n = 2
        ode_coefs_theta = np.array([
            [ # theta^0
                [ # x^0
                    [0, -1],
                    [-n**2, 0],
                ],
                [ # x^1
                    [0, 0],
                    [0, 0],
                ],
                [ # x^2
                    [0, 0],
                    [1, 0],
                ],
            ],
            [ # theta^1
                [ # x^0
                    [1, 0],
                    [0, 1],
                ],
                [ # x^1
                    [0, 0],
                    [0, 0],
                ],
                [ # x^2
                    [0, 0],
                    [0, 0],
                ],
            ],
        ])
        s = solve(ode_coefs_theta, min_terms=5)
        self.assertEqual(len(s), 2)

        lj, gj = s[0]
        self.assertAlmostEqual(lj, -2)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g / g[0, 0, 0, 0], [
            [[[1], [-2]], [[0], [0]]],
            [[[0], [0]], [[0], [0]]],
            [[[1/4], [0]], [[0], [0]]],
            [[[0], [0]], [[0], [0]]],
            [[[1/64], [-1/32]], [[-1/16], [-1/8]]],
            [[[0], [0]], [[0], [0]]],
            [[[-11/2304], [-1/72]], [[1/192], [1/48]]],
            [[[0], [0]], [[0], [0]]],
            [[[31/147456], [9/8192]], [[-1/6144], [-1/1024]]],
        ], atol=1e-10)

        lj, gj = s[1]
        self.assertAlmostEqual(lj, 2)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g / g[0, 0, 0, 0], [
            [[[1], [2]]],
            [[[0], [0]]],
            [[[-1/12], [-1/3]]],
            [[[0], [0]]],
            [[[1/384], [1/64]]],
        ], atol=1e-10)

    
    def test_spherical_bessel(self):
        # nth spherical Bessel equation converted to a system of two 1st order ODEs
        # f = (y, x * y')
        n = 2
        ode_coefs_theta = np.array([
            [ # theta^0
                [ # x^0
                    [0, -1],
                    [-n * (n + 1), 1],
                ],
                [ # x^1
                    [0, 0],
                    [0, 0],
                ],
                [ # x^2
                    [0, 0],
                    [1, 0],
                ],
            ],
            [ # theta^1
                [ # x^0
                    [1, 0],
                    [0, 1],
                ],
                [ # x^1
                    [0, 0],
                    [0, 0],
                ],
                [ # x^2
                    [0, 0],
                    [0, 0],
                ],
            ],
        ])
        s = solve(ode_coefs_theta, min_terms=5)
        self.assertEqual(len(s), 2)

        lj, gj = s[0]
        self.assertAlmostEqual(lj, -3)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g * (-3 / g[0, 0, 0, 0]), [
            [[[-3], [9]]],
            [[[0], [0]]],
            [[[-1/2], [1/2]]],
            [[[0], [0]]],
            [[[-1/8], [-1/8]]],
            [[[0], [0]]],
            [[[1/48], [1/16]]],
            [[[0], [0]]],
            [[[-1/1152], [-5/1152]]],
            [[[0], [0]]],
        ])

        lj, gj = s[1]
        self.assertAlmostEqual(lj, 2)
        self.assertEqual(len(gj), 1)
        self.assertEqual(len(gj[0]), 1)
        g = gj[0][0]
        np.testing.assert_allclose(g / g[0, 0, 0, 0], [
            [[[1], [2]]],
            [[[0], [0]]],
            [[[-1/14], [-4/14]]],
            [[[0], [0]]],
            [[[1/504], [6/504]]],
        ])


    def test_barkatou(self):
        m = 3
        a = 5
        ode_coefs_theta = np.array([
            [ # theta^0
                [[0, 2, 0], [0, 0, 0], [0, 0, 0]], # x^0
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^1
                [[-m**3, 0, 0], [0, m, 0], [0, 0, -m**2]], # x^2
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^3
                [[0, 0, 0], [0, 0, a], [0, 0, 0]], # x^4
            ],
            [ # theta^1
                [[-2*m, -1, 0], [-4, 0, 0], [-1, 0, 0]], # x^0
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^1
                [[0, 0, 0], [m**2, 0, 0], [0, 0, 0]], # x^2
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^3
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^4
            ],
            [ # theta^2
                [[m, 0, 0], [4, 0, 0], [0, 0, 1]], # x^0
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^1
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^2
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^3
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^4
            ],
            [ # theta^3
                [[0, 0, 0], [-1, 0, 0], [0, 0, 0]], # x^0
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^1
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^2
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^3
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]], # x^4
            ],
        ])
        s = solve(ode_coefs_theta, min_terms=3, lambda_roots=[0, 2])
        self.assertEqual(len(s), 2)

        lj, gj = s[0]
        self.assertAlmostEqual(lj, 0)
        self.assertEqual(len(gj), 2)
        self.assertEqual(len(gj[0]), 1)
        self.assertEqual(len(gj[1]), 2)

        g = gj[0][0]
        np.testing.assert_allclose(g / g[0, 0, 0, 0], [
            [[[1], [0], [0]], [[0], [0], [0]]],
            [[[0], [0], [0]], [[0], [0], [0]]],
            [[[-9/4], [0], [-9/4]], [[9/2], [0], [9/4]]],
            [[[0], [0], [0]], [[0], [0], [0]]],
            [[[-405/64], [0], [-243/64]], [[81/16], [0], [81/32]]],
        ], atol=1e-10)
        g = gj[1][0]
        np.testing.assert_allclose(g / g[0, 0, 2, 0], [
            [[[0], [0], [1]]],
            [[[0], [0], [0]]],
            [[[0], [0], [9/4]]],
            [[[0], [0], [0]]],
            [[[5/16], [15/4], [43/32]]],
        ], atol=1e-10)
        g = gj[1][1]
        np.testing.assert_allclose(g / g[0, 1, 2, 0], [
            [[[0], [0], [0]], [[0], [0], [1]]],
            [[[0], [0], [0]], [[0], [0], [0]]],
            [[[0], [0], [-9/4]], [[0], [0], [9/4]]],
            [[[0], [0], [0]], [[0], [0], [0]]],
            [[[-25/64], [-15/4], [-129/64]], [[5/16], [15/4], [43/32]]],
        ], atol=1e-10)

        lj, gj = s[1]
        self.assertAlmostEqual(lj, 2)
        self.assertEqual(len(gj), 2)
        self.assertEqual(len(gj[0]), 1)
        self.assertEqual(len(gj[1]), 2)

        g = gj[0][0]
        np.testing.assert_allclose(g / g[0, 0, 1, 0], [
            [[[0], [1], [0]]],
            [[[0], [0], [0]]],
            [[[3/16], [9/4], [3/64]]],
        ], atol=1e-10)
        g = gj[1][0]
        np.testing.assert_allclose(g / g[0, 0, 2, 0], [
            [[[2], [12], [1]]],
            [[[0], [0], [0]]],
            [[[9/2], [27], [27/16]]],
        ], atol=1e-10)
        g = gj[1][1]
        np.testing.assert_allclose(g / g[0, 0, 0, 0], [
            [[[1], [0], [0]], [[2], [12], [1]]],
            [[[0], [0], [0]], [[0], [0], [0]]],
            [[[-27/8], [-27], [-45/32]], [[9/2], [27], [27/16]]],
        ], atol=1e-10)

import unittest
import numpy as np


from frobenius.apoly import ArrayPoly

class Test(unittest.TestCase):
    
    def test_coefs(self):
        coefs = \
            np.array([10, 20, 30])[:, np.newaxis, np.newaxis] + \
            np.array([[1, 2], [3, 4]])
        a = ArrayPoly(coefs)
        np.testing.assert_allclose(a.coefs, coefs)

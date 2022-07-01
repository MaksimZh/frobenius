import numpy as np


class ArrayPoly:

    def __init__(self, coefs):
        self.coefs = coefs

    @property
    def ndim(self):
        return self.coefs.ndim - 1

    @property
    def npow(self):
        return self.coefs.shape[0]

    @property
    def shape(self):
        return self.coefs.shape[1:]

    def __call__(self, x):
        pows = np.arange(self.npow)
        xPows = (x ** pows)[(slice(None),) + (np.newaxis,) * self.ndim]
        return np.sum(self.coefs * xPows, axis=0)

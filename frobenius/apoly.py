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
        a = (slice(None),)
        na = (np.newaxis,)
        ell = (Ellipsis,)
        if isinstance(x, np.ndarray):
            cs = a + na * x.ndim + ell
            xPows = x[na + ell + na * self.ndim] ** pows[a + na * (x.ndim + self.ndim)]
        else:
            cs = ()
            xPows = x ** pows[a + na * self.ndim]
        return np.sum(self.coefs[cs] * xPows, axis=0)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return ArrayPoly(self.coefs[(slice(None),) + index])
        else:
            return ArrayPoly(self.coefs[:, index])

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

    def __coefsIndex(self, index):
        if isinstance(index, tuple):
            return (slice(None),) + index
        else:
            return (slice(None), index)

    def __getitem__(self, index):
        return ArrayPoly(self.coefs[self.__coefsIndex(index)])

    def __setitem__(self, index, value):
        if value.npow > self.npow:
            coefs = np.zeros((value.npow,) + self.shape,
                dtype=self.coefs.dtype)
            coefs[:self.npow] = self.coefs
            self.coefs = coefs
        self.coefs[self.__coefsIndex(index)][:value.npow] = value.coefs

    def __pos__(self):
        return self

    def __neg__(self):
        return ArrayPoly(-self.coefs)

    def __mul__(self, value):
        return ArrayPoly(self.coefs * value)

    def __rmul__(self, value):
        return ArrayPoly(value * self.coefs)

    def __truediv__(self, value):
        return ArrayPoly(self.coefs / value)

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
        si = (np.newaxis,) * self.ndim
        if isinstance(x, np.ndarray):
            xi = (np.newaxis,) * x.ndim
            return np.sum(
                    self.coefs[(slice(None),) + xi + (Ellipsis,)] * \
                    x[(np.newaxis, Ellipsis) + si] ** \
                    pows[(slice(None),) + xi + si],
                axis=0)
        else:
            return np.sum(self.coefs * x ** pows[(slice(None),) + si], axis=0)

    def __coefsIndex(self, index):
        if isinstance(index, tuple):
            return (slice(None),) + index
        else:
            return (slice(None), index)

    def __getitem__(self, index):
        return ArrayPoly(self.coefs[self.__coefsIndex(index)])

    def __setitem__(self, index, value):
        if value.npow > self.npow:
            coefs = np.zeros((value.npow, *self.shape), dtype=self.coefs.dtype)
            coefs[:self.npow] = self.coefs
            self.coefs = coefs
        self.coefs[self.__coefsIndex(index)][:value.npow] = value.coefs

    def __pos__(self):
        return self

    def __neg__(self):
        return ArrayPoly(-self.coefs)

    def __mul__(self, value):
        if not isinstance(value, ArrayPoly):
            return ArrayPoly(self.coefs * value)
        npow = self.npow + value.npow - 1
        coefs = np.zeros((npow, *self.shape),
            dtype=np.result_type(self.coefs, value.coefs))
        for s in range(self.npow):
            coefs[s : s + value.npow] += self.coefs[s : s + 1] * value.coefs
        return ArrayPoly(coefs)

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, value):
        return ArrayPoly(self.coefs / value)

    def __floordiv__(self, value):
        return ArrayPoly(self.coefs // value)

    def __add__(self, value):
        npow = max(self.npow, value.npow)
        coefs = np.zeros((npow, *self.shape),
            dtype=np.result_type(self.coefs, value.coefs))
        coefs[:self.npow] += self.coefs
        coefs[:value.npow] += value.coefs
        return ArrayPoly(coefs)

    def __sub__(self, value):
        npow = max(self.npow, value.npow)
        coefs = np.zeros((npow, *self.shape),
            dtype=np.result_type(self.coefs, value.coefs))
        coefs[:self.npow] += self.coefs
        coefs[:value.npow] -= value.coefs
        return ArrayPoly(coefs)

    def __matmul__(self, value):
        assert(self.ndim >= 2)
        assert(value.ndim >= 2)
        npow = self.npow + value.npow - 1
        bc = np.broadcast(self.coefs[0, ..., 0:1], value.coefs[0, ..., 0:1, :])
        coefs = np.zeros((npow, *bc.shape),
            dtype=np.result_type(self.coefs, value.coefs))
        for s in range(self.npow):
            coefs[s : s + value.npow] += self.coefs[s : s + 1] @ value.coefs
        return ArrayPoly(coefs)

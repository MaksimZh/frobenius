import numpy as np


class ArrayPoly:

    def __init__(self, coefs):
        if not isinstance(coefs, np.ndarray):
            coefs = np.array(coefs)
        if coefs.shape == ():
            coefs = coefs.reshape(1)
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

    def __call__(self, x, deriv=0):
        if isinstance(x, np.ndarray):
            return self.__evalArray(x, deriv)
        elif isinstance(x, ArrayPoly):
            assert(deriv == 0)
            return self.__substPoly(x)
        else:
            return self.__evalScalar(x, deriv)

    def __getitem__(self, index):
        return ArrayPoly(self.coefs[self.__coefsIndex(index)])

    def __setitem__(self, index, value):
        if value.npow > self.npow:
            coefs = np.zeros((value.npow, *self.shape), dtype=self.coefs.dtype)
            coefs[:self.npow] = self.coefs
            self.coefs = coefs
        self.coefs[self.__coefsIndex(index)][:value.npow] = value.coefs
        if value.npow < self.npow:
            self.coefs[self.__coefsIndex(index)][value.npow:] = 0

    def __pos__(self):
        return self

    def __neg__(self):
        return ArrayPoly(-self.coefs)

    def __mul__(self, value):
        if not isinstance(value, ArrayPoly):
            return ArrayPoly(self.coefs * value)
        npow = self.npow + value.npow - 1
        result_shape = np.broadcast(self.coefs[0], value.coefs[0]).shape
        coefs = np.zeros((npow, *result_shape),
            dtype=np.result_type(self.coefs, value.coefs))
        for s in range(self.npow):
            coefs[s : s + value.npow] += self.coefs[s : s + 1] * value.coefs
        return ArrayPoly(coefs)

    def __rmul__(self, value):
        return self * value

    def __truediv__(self, value):
        return ArrayPoly(self.coefs / value)

    def __floordiv__(self, value):
        if isinstance(value, ArrayPoly):
            q, _ = divmod(self, value)
            return q
        else:
            return ArrayPoly(self.coefs // value)

    def __mod__(self, value):
        if not isinstance(value, ArrayPoly):
            return NotImplemented
        _, r = divmod(self, value)
        return r


    def __add__(self, value):
        npow = max(self.npow, value.npow)
        result_shape = np.broadcast(self.coefs[0], value.coefs[0]).shape
        coefs = np.zeros((npow, *result_shape),
            dtype=np.result_type(self.coefs, value.coefs))
        coefs[:self.npow] += self.coefs
        coefs[:value.npow] += value.coefs
        return ArrayPoly(coefs)

    def __sub__(self, value):
        npow = max(self.npow, value.npow)
        result_shape = np.broadcast(self.coefs[0], value.coefs[0]).shape
        coefs = np.zeros((npow, *result_shape),
            dtype=np.result_type(self.coefs, value.coefs))
        coefs[:self.npow] += self.coefs
        coefs[:value.npow] -= value.coefs
        return ArrayPoly(coefs)

    def __matmul__(self, value):
        assert(self.ndim >= 2)
        assert(value.ndim >= 2)
        npow = self.npow + value.npow - 1
        result_shape = np.broadcast(
            self.coefs[0, ..., 0:1],
            value.coefs[0, ..., 0:1, :]).shape
        coefs = np.zeros((npow, *result_shape),
            dtype=np.result_type(self.coefs, value.coefs))
        for s in range(self.npow):
            coefs[s : s + value.npow] += self.coefs[s : s + 1] @ value.coefs
        return ArrayPoly(coefs)

    def __pow__(self, value):
        if type(value) != int:
            return NotImplemented
        result = ArrayPoly(np.ones_like(self.coefs[:1]))
        for i in range(value):
            result *= self
        return result

    def __divmod__(self, value):
        if value.ndim > 0:
            return NotImplemented
        remainder_poly_coefs = self.coefs / np.ones_like(value.coefs[:1])[0]
        divisor_poly_coefs = value.coefs[_it((), self.ndim)]
        quotient_poly_coefs = []
        si = self.npow - value.npow
        sf = self.npow - 1
        for _ in range(self.npow - value.npow + 1):
            qc = remainder_poly_coefs[sf] / divisor_poly_coefs[-1]
            remainder_poly_coefs[si : sf + 1] -= qc * divisor_poly_coefs
            si -= 1
            sf -= 1
            quotient_poly_coefs.append(qc)
        rem = ArrayPoly(remainder_poly_coefs[: value.npow - 1])
        quo = ArrayPoly(np.array(list(reversed(quotient_poly_coefs))))
        return quo, rem

    def __coefsIndex(self, index):
        if isinstance(index, tuple):
            return (slice(None),) + index
        else:
            return (slice(None), index)

    def __evalScalar(self, x, deriv):
        pows = np.arange(self.npow)
        factors = np.ones(self.npow)
        for i in range(deriv):
            factors[i:] *= pows[: len(pows) - i]
        pows -= deriv
        pows[pows < 0] = 0
        it = _it((), self.ndim)
        return np.sum(
            self.coefs * factors[it] * x ** pows[it],
            axis=0)

    def __evalArray(self, x, deriv):
        pows = np.arange(self.npow)
        factors = np.ones(self.npow)
        for i in range(deriv):
            factors[i:] *= pows[: len(pows) - i]
        pows -= deriv
        pows[pows < 0] = 0
        return np.sum(
                self.coefs[_it((), x.ndim, -1)] * \
                factors[_it((), x.ndim, self.ndim)] * \
                x[_it(1, -1, self.ndim)] ** \
                    pows[_it((), x.ndim, self.ndim)],
            axis=0)

    def __substPoly(self, x):
        if x.ndim == 0:
            x = ArrayPoly(x.coefs[_it((), self.ndim)])
        result = ArrayPoly(np.zeros_like([self.coefs[0] * x.coefs[0]]))
        for p in range(self.npow):
            s = ArrayPoly(self.coefs[p : p + 1])
            result += s * x ** p
        return result


def trim(a, atol=1e-12):
    coefs = a.coefs
    while len(coefs) > 1 and np.max(np.abs(coefs[-1])) < atol:
        coefs = coefs[:-1]
    return ArrayPoly(coefs)


def det(a):
    assert(a.ndim >= 2)
    assert(a.shape[-1] == a.shape[-2])
    column_locked = [False] * a.shape[-1]

    def minor(i):
        if i == a.shape[-2]:
            return 1
        factor = 1
        result = ArrayPoly(0)
        for j in range(a.shape[-1]):
            if not column_locked[j]:
                column_locked[j] = True
                result = result + factor * a[..., i, j] * minor(i + 1)
                factor *= -1
                column_locked[j] = False
        return result

    return minor(0)

def _it(*args):
    result = ()
    for a in args:
        if a == ():
            result += (slice(None),)
        elif a >= 0:
            result += (np.newaxis,) * a
        else:
            result += (Ellipsis,)
    return result

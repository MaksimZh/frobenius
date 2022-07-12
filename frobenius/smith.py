import numpy as np

from frobenius.apoly import ArrayPoly, trim


def smith(a, p):
    n = a.shape[0]
    k = 0
    i = 0
    x = ArrayPoly(np.eye(n, dtype=np.common_type(a.coefs, p.coefs))[np.newaxis])
    z = ArrayPoly(np.zeros_like(x.coefs))
    pp = [ArrayPoly(1)]
    kappa = np.zeros(n, dtype=int)
    while i < n:
        for j in range(0, n - i):
            z[:, i : i + 1] = a @ x[:, i : i + 1] // pp[k] % p
            c = expandLast(z[:, : i + 1])
            if c is None:
                kappa[i] = k
                i += 1
            else:
                for m in range(0, i):
                    x[:, i] = x[:, i] - pp[k - kappa[m]] * c[m] * x[:, m]
                x.coefs[..., i:] = np.roll(x.coefs[..., i:], shift=-1, axis=-1)
        k += 1
        pp.append(pp[-1] * p)
    y = a @ x
    for i in range(n):
        y[:, i] //= pp[kappa[i]]
    return trim(x), trim(y), kappa


def expandLast(z):
    a = z.coefs.reshape(-1, z.shape[-1])
    u, s, vh = np.linalg.svd(a)
    if s[-1] > 1e-12:
        return None
    else:
        z = np.conj(vh[-1])
    c = -z[:-1] / z[-1]
    return c

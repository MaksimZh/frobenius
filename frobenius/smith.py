import numpy as np

from frobenius.apoly import ArrayPoly


def smith(a, p):
    n = a.shape[0]
    k = 0
    i = 0
    x = ArrayPoly(np.eye(n)[np.newaxis])
    y = ArrayPoly(np.zeros_like(x.coefs))
    pp = [ArrayPoly(np.array([1]))]
    kappa = np.zeros(n, dtype=int)
    while i < n:
        for j in range(0, n - i):
            y[:, i] = a @ x[:, i : i + 1] // pp[k] % p
            c = expand(y[:, i], y[:, :i])
            if c is None:
                kappa[i] = k
                i += 1
            else:
                for m in range(0, i):
                    x[:, i] = x[:, i] - pp[k - kappa[m]] * c[m] * x[:, m]
                x.coefs = np.roll(x.coefs, shift=-1, axis=-1)
        k += 1
        pp.append(pp[-1] * p)
    return x, y, kappa


def expand(v, basis):
    npow = max(v.npow, basis.npow)
    vc = np.zeros((npow, *v.shape), dtype=complex)
    vc[:v.npow] = v.coefs
    bc = np.zeros((npow, *basis.shape), dtype=complex)
    bc[:basis.npow] = basis.coefs
    c = np.sum(np.conj(bc) * vc[..., np.newaxis], axis=(0, 1))
    rc = vc - np.sum(bc * c, axis=-1)
    if np.sum(np.abs(rc)) < 1e-12:
        return c
    else:
        return None

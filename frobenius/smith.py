import numpy as np

from frobenius.apoly import ArrayPoly, trim


def smith(a, factor, atol=1e-12):
    n = a.shape[0]
    exponent = 0
    column = 0
    inv_right_matrix = ArrayPoly(np.eye(n, dtype=np.common_type(a.coefs, factor.coefs))[np.newaxis])
    remainder_matrix = ArrayPoly(np.zeros_like(inv_right_matrix.coefs))
    factor_powers = [ArrayPoly(1)]
    diag_factor_exponents = np.zeros(n, dtype=int)
    while column < n:
        for j in range(0, n - column):
            remainder_matrix[:, column : column + 1] = \
                a @ inv_right_matrix[:, column : column + 1] // factor_powers[exponent] % factor
            expansion_coefs = expandLast(remainder_matrix[:, : column + 1], atol)
            if expansion_coefs is None:
                diag_factor_exponents[column] = exponent
                column += 1
            else:
                for m in range(0, column):
                    inv_right_matrix[:, column] = \
                        inv_right_matrix[:, column] - \
                        factor_powers[exponent - diag_factor_exponents[m]] * \
                        expansion_coefs[m] * \
                        inv_right_matrix[:, m]
                inv_right_matrix.coefs[..., column:] = \
                    np.roll(inv_right_matrix.coefs[..., column:], shift=-1, axis=-1)
        exponent += 1
        factor_powers.append(factor_powers[-1] * factor)
    left_matrix = a @ inv_right_matrix
    for column in range(n):
        left_matrix[:, column] //= factor_powers[diag_factor_exponents[column]]
    return trim(inv_right_matrix, atol), trim(left_matrix, atol), diag_factor_exponents


def expandLast(z, atol=1e-12):
    a = z.coefs.reshape(-1, z.shape[-1])
    u, s, vh = np.linalg.svd(a)
    if s[-1] > atol:
        return None
    else:
        z = np.conj(vh[-1])
    c = -z[:-1] / z[-1]
    return c

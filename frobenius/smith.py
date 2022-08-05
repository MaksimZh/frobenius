import numpy as np

from frobenius.apoly import ArrayPoly, trim


def smith(a, factor, atol=1e-12):
    size = a.shape[0]
    inv_right_matrix = ArrayPoly(np.eye(size, dtype=np.common_type(a.coefs, factor.coefs))[np.newaxis])
    remainder_matrix = ArrayPoly(np.zeros_like(inv_right_matrix.coefs))
    factor_powers = [ArrayPoly(1)]
    diag_factor_exponents = np.zeros(size, dtype=int)
    column = 0
    exponent = 0
    while column < size:
        for j in range(0, size - column):
            remainder_matrix[:, column : column + 1] = \
                a @ inv_right_matrix[:, column : column + 1] // factor_powers[exponent] % factor
            expansion_coefs = expandLast(remainder_matrix[:, : column + 1], atol)
            is_lin_indep_columns = expansion_coefs is None
            if is_lin_indep_columns:
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
    del column, exponent

    left_matrix = a @ inv_right_matrix
    # multiply by inverse diagonal part of local normal Smith form
    for column in range(size):
        left_matrix[:, column] //= factor_powers[diag_factor_exponents[column]]
    return trim(inv_right_matrix, atol), trim(left_matrix, atol), diag_factor_exponents


def expandLast(matrix_poly, atol=1e-12):
    coef_columns = matrix_poly.coefs.reshape(-1, matrix_poly.shape[-1])
    u, singular_values, inv_right_matrix = np.linalg.svd(coef_columns)
    del u
    is_lin_indep_columns = singular_values[-1] > atol
    del singular_values
    if is_lin_indep_columns:
        return None
    else:
        matrix_poly = np.conj(inv_right_matrix[-1])
    coef_columns = -matrix_poly[:-1] / matrix_poly[-1]
    return coef_columns

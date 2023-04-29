import numpy as np
from typing import Any

from frobenius.apoly import ArrayPoly, trim
from frobenius.tools import Status, status


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
            # subsequent // and % operations is essential part of the algorithm
            # we extract the terms with specific power of factor
            column_vector = a @ inv_right_matrix[:, column : column + 1]
            higher_pow_coef = column_vector // factor_powers[exponent]
            remainder_matrix[:, column : column + 1] = higher_pow_coef % factor
            del column_vector, higher_pow_coef
            expansion_coefs = expandLast(remainder_matrix[:, : column + 1], atol)
            is_lin_indep_columns = expansion_coefs is None
            if is_lin_indep_columns:
                diag_factor_exponents[column] = exponent
                column += 1
            else:
                for m in range(0, column):
                    inv_right_matrix[:, column] -= \
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

# Collects columns added to the right side
class ColumnCollector(Status):

    _nrows: int
    _data: list[ArrayPoly]
    
    # CONSTRUCTOR
    # Create empty collector
    def __init__(self, nrows: int) -> None:
        super().__init__()
        self._nrows = nrows
        self._data = []


    # COMMANDS

    # Add column to the right
    # PRE: `column` is Nx1 polynomial matrix
    # PRE: `column` height matches `nrows`
    # POST: `column` is added to the right end of the queue
    @status("OK", "NOT_COLUMN", "SIZE_MISMATCH")
    def push_right(self, column: ArrayPoly) -> None:
        if column.ndim != 2 or column.shape[1] != 1:
            self._set_status("push_right", "NOT_COLUMN")
            return
        if column.shape[0] != self._nrows:
            self._set_status("push_right", "SIZE_MISMATCH")
            return
        self._data.append(column)
        self._set_status("push_right", "OK")


    # QUERIES

    # Check if queue is empty
    def is_empty(self) -> bool:
        return len(self._data) == 0


class ColumnQueue(ColumnCollector):

    # CONSTRUCTOR
    # Create empty queue
    def __init__(self, nrows: int) -> None:
        super().__init__(nrows)

    
    # COMMANDS

    # Remove the left column 
    # PRE: queue is not empty
    # POST: left column removed
    @status("OK", "EMPTY")
    def pop_left(self) -> None:
        if self.is_empty():
            self._set_status("pop_left", "EMPTY")
            return
        self._data.pop(0)
        self._set_status("pop_left", "OK")

    
    # QUERIES

    # Get left column
    # PRE: queue is not empty
    @status("OK", "EMPTY")
    def get_left(self) -> ArrayPoly:
        if self.is_empty():
            self._set_status("get_left", "EMPTY")
            return ArrayPoly(0)
        self._set_status("get_left", "OK")
        assert len(self._data) > 0
        return self._data[0]


class ExtendibleMatrix(ColumnCollector):

    # CONSTRUCTOR
    # Create empty matrix
    def __init__(self, nrows: int) -> None:
        super().__init__(nrows)


    # QUERIES
    
    # Get columns as matrix
    # PRE: collector is not empty
    @status("OK", "EMPTY")
    def get_matrix(self) -> ArrayPoly:
        if self.is_empty():
            self._set_status("get_matrix", "EMPTY")
            return ArrayPoly(0)
        self._set_status("get_matrix", "OK")
        max_npow = max(d.npow for d in self._data)
        ncols = len(self._data)
        common_type = np.common_type(*[d.coefs for d in self._data])
        result = ArrayPoly(np.zeros((max_npow, self._nrows, ncols), dtype=common_type))
        for i in range(ncols):
            result[:, i : i + 1] = self._data[i]
        return result

    # Expand given column over matrix columns
    # PRE: matrix is not empty
    # PRE: `column` size fits matrix
    # PRE: `column` is complanar with matrix columns
    @status("OK", "INVALID_COLUMN", "EMPTY", "NOT_COMPLANAR")
    def expand_column(self, column: ArrayPoly, atol: float = 1e-12) -> list[complex]:
        if column.ndim != 2 or column.shape[1] != 1 or column.shape[0] != self._nrows:
            self._set_status("expand_column", "INVALID_COLUMN")
            return []
        if self.is_empty():
            self._set_status("expand_column", "EMPTY")
            return []
        max_npow = max(d.npow for d in self._data)
        max_npow = max(max_npow, column.npow)
        ncols = len(self._data)
        coefs = np.zeros((max_npow * self._nrows, ncols + 1))
        for i in range(ncols):
            c = self._data[i].coefs.reshape(-1)
            coefs[:len(c), i] = c
        c = column.coefs.reshape(-1)
        coefs[:len(c), -1] = c
        u, s, vh = np.linalg.svd(coefs)
        if s[-1] > atol:
            self._set_status("expand_column", "NOT_COMPLANAR")
            return []
        self._set_status("expand_column", "OK")
        fitting_coefs = np.conj(vh[-1])
        expansion_coefs = -fitting_coefs[:-1] / fitting_coefs[-1]
        return list(expansion_coefs)

    # Make linear combination of columns with given coefficients
    # PRE: matrix is not empty
    # PRE: `factors` size fits number of columns
    @status("OK", "EMPTY", "SIZE_MISMATCH")
    def combine_columns(self, factors: list[Any]) -> ArrayPoly:
        if self.is_empty():
            self._set_status("combine_columns", "EMPTY")
            return ArrayPoly(0)
        if len(factors) != len(self._data):
            self._set_status("combine_columns", "SIZE_MISMATCH")
            return ArrayPoly(0)
        self._set_status("combine_columns", "OK")
        return sum((f * c for f, c in zip(factors, self._data)),
                   start=ArrayPoly([[[0]] * self._nrows]))

import numpy as np
from typing import Any

from frobenius.apoly import ArrayPoly, trim
from frobenius.tools import Status, status


def smith(a, factor, atol=1e-12):
    size = a.shape[0]
    inv_right_matrix = ExtendibleMatrix(size)
    remainder_matrix = ExtendibleMatrix(size)
    seed_matrix = ArrayPoly(np.eye(size, dtype=np.common_type(a.coefs, factor.coefs))[np.newaxis])
    seed_columns = ColumnQueue(size)
    for i in range(size):
        seed_columns.push_right(seed_matrix[:, i : i + 1])
    factor_powers = [ArrayPoly(1)]
    diag_factor_exponents = []
    exponent = 0
    new_seed_columns = ColumnQueue(size)
    while not (seed_columns.is_empty() and new_seed_columns.is_empty()):
        if seed_columns.is_empty():
            exponent += 1
            factor_powers.append(factor_powers[-1] * factor)
            seed_columns, new_seed_columns = new_seed_columns, seed_columns
        seed_column = seed_columns.get_left()
        seed_columns.pop_left()
        # subsequent // and % operations is essential part of the algorithm
        # we extract the terms with specific power of factor
        remainder_column = (a @ seed_column) // factor_powers[exponent] % factor
        if is_zero(remainder_column, atol):
            new_seed_columns.push_right(seed_column)
            continue
        expansion_coefs = remainder_matrix.expand_column(remainder_column)
        if remainder_matrix.is_status("expand_column", "EMPTY") \
                or remainder_matrix.is_status("expand_column", "NOT_COMPLANAR"):
            inv_right_matrix.push_right(seed_column)
            remainder_matrix.push_right(remainder_column)
            diag_factor_exponents.append(exponent)
            continue
        assert remainder_matrix.is_status("expand_column", "OK")
        assert not inv_right_matrix.is_empty()
        expansion_factor_powers = [factor_powers[exponent - dfe] for dfe in diag_factor_exponents]
        coefs = [ec * efp for ec, efp in zip(expansion_coefs, expansion_factor_powers)]
        new_seed = remainder_column - inv_right_matrix.combine_columns(coefs)
        new_seed_columns.push_right(new_seed)

    inv_right_matrix = inv_right_matrix.get_matrix()
    left_matrix = a @ inv_right_matrix
    # multiply by inverse diagonal part of local normal Smith form
    for column in range(size):
        left_matrix[:, column] //= factor_powers[diag_factor_exponents[column]]
    return trim(inv_right_matrix, atol), trim(left_matrix, atol), np.array(diag_factor_exponents)


def is_zero(poly: ArrayPoly, atol: float = 1e-12) -> bool:
    return np.sum(np.abs(poly.coefs)) < atol

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

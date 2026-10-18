from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .config import MatrixDiagnostics, ScanPoint, SolverContext
from .hh_coefficients import HHMatrixCoefficients, compute_hh_coefficients


def assemble_hh_matrix(
    ctx: SolverContext,
    point: ScanPoint,
    *,
    quadrature_order: int = 120,
    return_diagnostics: bool = False,
) -> csr_matrix | tuple[csr_matrix, MatrixDiagnostics]:
    coeff = compute_hh_coefficients(ctx, point, quadrature_order=quadrature_order)
    matrix = assemble_hh_matrix_from_coefficients(coeff)
    if return_diagnostics:
        return matrix, coeff.to_diagnostics()
    return matrix


def assemble_hh_matrix_from_coefficients(coeff: HHMatrixCoefficients) -> csr_matrix:
    coeff.validate()

    if coeff.iem == 2:
        raise NotImplementedError(
            "Darwin generalized eigenvalue matrix is out of scope for this milestone."
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for snj in range(coeff.snj):
        jjx = snj
        jjy = snj + coeff.snj1
        jjz = snj + 2 * coeff.snj1
        _extend(
            rows,
            cols,
            data,
            jjx,
            [jjx, coeff.snj3, coeff.snj3 + 1, coeff.snj3 + 2],
            [coeff.csnj[snj], coeff.bsnj[snj, 0, 0], coeff.bsnj[snj, 0, 1], coeff.bsnj[snj, 0, 2]],
        )
        _extend(
            rows,
            cols,
            data,
            jjy,
            [jjy, coeff.snj3, coeff.snj3 + 1, coeff.snj3 + 2],
            [coeff.csnj[snj], coeff.bsnj[snj, 1, 0], coeff.bsnj[snj, 1, 1], coeff.bsnj[snj, 1, 2]],
        )
        _extend(
            rows,
            cols,
            data,
            jjz,
            [jjz, coeff.snj3, coeff.snj3 + 1, coeff.snj3 + 2],
            [coeff.csnj[snj], coeff.bsnj[snj, 2, 0], coeff.bsnj[snj, 2, 1], coeff.bsnj[snj, 2, 2]],
        )

    _extend_many(rows, cols, data, coeff.snj3, range(0, coeff.snj1), -1.0)
    _extend_many(rows, cols, data, coeff.snj3 + 1, range(coeff.snj1, 2 * coeff.snj1), -1.0)
    _extend_many(rows, cols, data, coeff.snj3 + 2, range(2 * coeff.snj1, 3 * coeff.snj1), -1.0)

    for s in range(coeff.s_count):
        row_x = coeff.snj1 - coeff.s_count + s
        row_y = 2 * coeff.snj1 - coeff.s_count + s
        row_z = 3 * coeff.snj1 - coeff.s_count + s
        _extend(
            rows,
            cols,
            data,
            row_x,
            [coeff.snj3, coeff.snj3 + 1, coeff.snj3 + 2],
            coeff.bs[s, 0, :],
        )
        _extend(
            rows,
            cols,
            data,
            row_y,
            [coeff.snj3, coeff.snj3 + 1, coeff.snj3 + 2],
            coeff.bs[s, 1, :],
        )
        _extend(
            rows,
            cols,
            data,
            row_z,
            [coeff.snj3, coeff.snj3 + 1, coeff.snj3 + 2],
            coeff.bs[s, 2, :],
        )

    _extend(
        rows,
        cols,
        data,
        coeff.snj3,
        [coeff.snj3 + 4],
        [coeff.c2 * coeff.kz],
    )
    _extend(
        rows,
        cols,
        data,
        coeff.snj3 + 1,
        [coeff.snj3 + 3, coeff.snj3 + 5],
        [-coeff.c2 * coeff.kz, coeff.c2 * coeff.kx],
    )
    _extend(
        rows,
        cols,
        data,
        coeff.snj3 + 2,
        [coeff.snj3 + 4],
        [-coeff.c2 * coeff.kx],
    )

    _extend(rows, cols, data, coeff.snj3 + 3, [coeff.snj3 + 1], [-coeff.kz])
    _extend(
        rows,
        cols,
        data,
        coeff.snj3 + 4,
        [coeff.snj3, coeff.snj3 + 2],
        [coeff.kz, -coeff.kx],
    )
    _extend(rows, cols, data, coeff.snj3 + 5, [coeff.snj3 + 1], [coeff.kx])

    return coo_matrix((data, (rows, cols)), shape=(coeff.nn, coeff.nn), dtype=np.complex128).tocsr()


def _extend(rows: list[int], cols: list[int], data: list[complex], row: int, col_values, values) -> None:
    rows.extend([row] * len(col_values))
    cols.extend(col_values)
    data.extend(values)


def _extend_many(rows: list[int], cols: list[int], data: list[complex], row: int, col_values, value: complex) -> None:
    col_values = list(col_values)
    rows.extend([row] * len(col_values))
    cols.extend(col_values)
    data.extend([value] * len(col_values))

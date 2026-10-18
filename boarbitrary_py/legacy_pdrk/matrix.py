from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from .coefficients import (
    effective_em_wavevector,
    evaluate_em3d_coefficients,
    evaluate_es3d_coefficients,
)
from .config import LegacyCoefficients, LegacyContext, LegacyMatrix, LegacyScanPoint


def assemble_es3d_matrix(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    coefficients: LegacyCoefficients | None = None,
    quadrature_order: int = 120,
) -> LegacyMatrix:
    """Assemble the legacy ES3D matrix in MATLAB species/harmonic/pole order."""

    coeff = coefficients or evaluate_es3d_coefficients(
        ctx,
        point,
        quadrature_order=quadrature_order,
    )
    if coeff.es_residues is None:
        raise ValueError("ES3D assembly requires scalar residues")
    if coeff.poles.shape != (ctx.snj,) or coeff.es_residues.shape != (ctx.snj,):
        raise ValueError("Legacy ES3D coefficient shape does not match SNJ")

    states = np.arange(ctx.snj, dtype=np.int64)
    field = np.full(ctx.snj, ctx.snj, dtype=np.int64)
    rows = np.concatenate((states, states, field, np.asarray([ctx.snj], dtype=np.int64)))
    cols = np.concatenate((states, field, states, np.asarray([ctx.snj], dtype=np.int64)))
    values = np.concatenate(
        (
            coeff.poles,
            coeff.es_residues,
            -coeff.poles,
            np.asarray([-np.sum(coeff.es_residues)], dtype=np.complex128),
        )
    )
    matrix = coo_matrix((values, (rows, cols)), shape=(ctx.snj1, ctx.snj1)).tocsr()
    matrix.eliminate_zeros()
    return LegacyMatrix(matrix=matrix, mass_matrix=None, coefficients=coeff)


def _append_entry(
    rows: list[int],
    cols: list[int],
    values: list[complex],
    row: int,
    col: int,
    value: complex,
) -> None:
    rows.append(row)
    cols.append(col)
    values.append(value)


def _assemble_darwin_mass_matrix(ctx: LegacyContext, point: LegacyScanPoint):
    k, kx, kz = effective_em_wavevector(point)
    rows = list(range(ctx.nn))
    cols = list(range(ctx.nn))
    values: list[complex] = [1.0] * ctx.nn
    ex = ctx.snj3
    ey = ex + 1
    ez = ex + 2
    for row, col, value in (
        (ex, ex, kx**2 / k**2),
        (ez, ex, kx * kz / k**2),
        (ex, ez, kx**2 / k**2),
        (ez, ez, kz**2 / k**2),
        (ex, ex, -1.0),
        (ey, ey, -1.0),
        (ez, ez, -1.0),
    ):
        _append_entry(rows, cols, values, row, col, value)
    mass = coo_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, cols)),
        shape=(ctx.nn, ctx.nn),
    ).tocsr()
    mass.sum_duplicates()
    mass.eliminate_zeros()
    return mass


def assemble_em3d_matrix(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    coefficients: LegacyCoefficients | None = None,
    quadrature_order: int = 120,
) -> LegacyMatrix:
    """Assemble the legacy EM3D/Darwin current-field matrix in one COO pass."""

    coeff = coefficients or evaluate_em3d_coefficients(
        ctx,
        point,
        quadrature_order=quadrature_order,
    )
    if coeff.em_residues is None or coeff.em_constant is None:
        raise ValueError("EM3D assembly requires 3x3 residue and constant blocks")
    if coeff.poles.shape != (ctx.snj,) or coeff.em_residues.shape != (ctx.snj, 3, 3):
        raise ValueError("Legacy EM3D coefficient shape does not match SNJ")
    if coeff.em_constant.shape != (3, 3):
        raise ValueError("Legacy EM3D constant block must be 3x3")

    _, kx, kz = effective_em_wavevector(point)
    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    field_start = ctx.snj3

    for state in range(ctx.snj):
        for current_axis in range(3):
            row = state + current_axis * ctx.snj1
            _append_entry(rows, cols, values, row, row, coeff.poles[state])
            for field_axis in range(3):
                _append_entry(
                    rows,
                    cols,
                    values,
                    row,
                    field_start + field_axis,
                    coeff.em_residues[state, current_axis, field_axis],
                )

    for axis in range(3):
        field_row = field_start + axis
        block_start = axis * ctx.snj1
        for current_col in range(block_start, block_start + ctx.snj1):
            _append_entry(rows, cols, values, field_row, current_col, -1.0)

    for current_axis in range(3):
        current_row = (current_axis + 1) * ctx.snj1 - 1
        for field_axis in range(3):
            _append_entry(
                rows,
                cols,
                values,
                current_row,
                field_start + field_axis,
                coeff.em_constant[current_axis, field_axis],
            )

    c2 = ctx.constants.c2
    ex, ey, ez, bx, by, bz = range(field_start, field_start + 6)
    for row, col, value in (
        (ex, by, c2 * kz),
        (ey, bx, -c2 * kz),
        (ey, bz, c2 * kx),
        (ez, by, -c2 * kx),
        (bx, ey, -kz),
        (by, ex, kz),
        (by, ez, -kx),
        (bz, ey, kx),
    ):
        _append_entry(rows, cols, values, row, col, value)

    matrix = coo_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, cols)),
        shape=(ctx.nn, ctx.nn),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    mass = _assemble_darwin_mass_matrix(ctx, point) if ctx.case.model.iem == 2 else None
    return LegacyMatrix(matrix=matrix, mass_matrix=mass, coefficients=coeff)

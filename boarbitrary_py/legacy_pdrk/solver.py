from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np
from scipy.linalg import eigvals

from .config import LegacyContext, LegacyMatrix, LegacyScanPoint, LegacyScanResult
from .matrix import assemble_em3d_matrix, assemble_es3d_matrix
from .minimal import solve_minimal_point
from .scan import iter_legacy_scan_points


def assemble_legacy_matrix(ctx: LegacyContext, point: LegacyScanPoint) -> LegacyMatrix:
    if ctx.case.model.iem == 0:
        return assemble_es3d_matrix(ctx, point)
    if ctx.case.model.iem in {1, 2}:
        return assemble_em3d_matrix(ctx, point)
    raise ValueError(f"Unsupported legacy iem={ctx.case.model.iem}")


def solve_legacy_matrix(matrix: LegacyMatrix) -> np.ndarray:
    dense = matrix.matrix.toarray()
    if matrix.mass_matrix is None:
        return eigvals(dense, overwrite_a=True, check_finite=False)
    mass = matrix.mass_matrix.toarray()
    return eigvals(dense, mass, overwrite_a=True, check_finite=False)


def solve_legacy_point(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    backend: str = "full",
) -> np.ndarray:
    if backend == "full":
        return solve_legacy_matrix(assemble_legacy_matrix(ctx, point))
    if backend == "minimal":
        return solve_minimal_point(ctx, point).roots
    raise ValueError("backend must be 'full' or 'minimal'")


def solve_legacy_scan(
    ctx: LegacyContext,
    scan_indices: Iterable[int] | None = None,
    *,
    backend: str = "full",
) -> LegacyScanResult:
    points = tuple(iter_legacy_scan_points(ctx))
    if scan_indices is None:
        selected = np.arange(len(points), dtype=np.int64)
    else:
        selected = np.asarray(tuple(scan_indices), dtype=np.int64)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(points)):
            raise ValueError("scan_indices must be a one-dimensional in-range sequence")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("scan_indices must not contain duplicates")

    roots = np.empty((len(selected), ctx.nn), dtype=np.complex128)
    k = np.empty(len(selected), dtype=float)
    kx = np.empty(len(selected), dtype=float)
    kz = np.empty(len(selected), dtype=float)
    theta = np.empty(len(selected), dtype=float)
    reduced_sizes = np.full(len(selected), ctx.nn, dtype=np.int64)
    structural_counts = np.zeros(len(selected), dtype=np.int64)
    started = perf_counter()
    for output_index, scan_index in enumerate(selected):
        point = points[int(scan_index)]
        if backend == "minimal":
            spectrum = solve_minimal_point(ctx, point)
            roots[output_index] = spectrum.roots
            reduced_sizes[output_index] = spectrum.realization.certificate.minimal_size
            structural_counts[output_index] = spectrum.structural_poles.size
        elif backend == "full":
            roots[output_index] = solve_legacy_point(ctx, point, backend="full")
        else:
            raise ValueError("backend must be 'full' or 'minimal'")
        k[output_index] = point.k
        kx[output_index] = point.kx
        kz[output_index] = point.kz
        theta[output_index] = point.theta_deg
    elapsed = perf_counter() - started
    return LegacyScanResult(
        roots=roots,
        scan_indices=selected,
        k=k,
        kx=kx,
        kz=kz,
        theta_deg=theta,
        elapsed_seconds=elapsed,
        backend=backend,
        reduced_matrix_sizes=reduced_sizes,
        structural_root_counts=structural_counts,
    )

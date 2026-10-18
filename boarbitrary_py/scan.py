from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import numpy as np

from .config import ScanPoint, ScanResult, SolverContext
from .initialize import initialize_case
from .matrix_hh import assemble_hh_matrix
from .solver import solve_eigenvalues, sort_by_growth


def iter_scan_points(
    ctx: SolverContext,
    *,
    max_points: int | None = None,
    scan_indices: Iterable[int] | None = None,
) -> Iterator[ScanPoint]:
    for linear_index in _selected_scan_point_indices(ctx, max_points=max_points, scan_indices=scan_indices):
        jpa = linear_index // len(ctx.pb)
        jpb = linear_index % len(ctx.pb)
        pa_value = ctx.pa[jpa]
        pb_value = ctx.pb[jpb]

        par = np.array(ctx.setup.par, copy=True)
        par[ctx.setup.ipa - 1] = _scale_scan_value(pa_value, ctx.setup.iloga)
        if ctx.setup.ipa != ctx.setup.ipb:
            par[ctx.setup.ipb - 1] = _scale_scan_value(pb_value, ctx.setup.ilogb)

        k_norm = abs(par[0])
        theta_deg = par[1]
        if ctx.setup.ipa > 2 and ctx.setup.ipb > 2:
            kz_norm = par[2]
            kx_norm = par[3]
            kz = kz_norm * ctx.kn
            kx = kx_norm * ctx.kn
        else:
            k = k_norm * ctx.kn
            kz = np.cos(np.deg2rad(theta_deg)) * k
            kx = np.sin(np.deg2rad(theta_deg)) * k
            kz_norm = kz / ctx.kn
            kx_norm = kx / ctx.kn

        k = float(np.sqrt(kx * kx + kz * kz))
        theta_deg = float(np.angle(kz + 1j * kx, deg=True))
        yield ScanPoint(
            jpa=jpa,
            jpb=jpb,
            pa_value=float(pa_value),
            pb_value=float(pb_value),
            par=par,
            k_norm=k / ctx.kn,
            theta_deg=theta_deg,
            kz_norm=kz / ctx.kn,
            kx_norm=kx / ctx.kn,
            k=k,
            kz=float(kz),
            kx=float(kx),
        )


def run_scan(
    ctx: SolverContext | None = None,
    *,
    max_points: int | None = None,
    scan_indices: Iterable[int] | None = None,
    quadrature_order: int = 120,
) -> ScanResult:
    ctx = ctx or initialize_case()
    if ctx.setup.sp != 0:
        raise NotImplementedError("The first HH milestone only supports the default sp=0 scan.")

    selected_indices = _selected_scan_point_indices(ctx, max_points=max_points, scan_indices=scan_indices)
    total_points = len(selected_indices)
    ww = np.full((total_points, ctx.nw), np.nan + 0j, dtype=np.complex128)
    kxx = np.zeros(total_points, dtype=float)
    kzz = np.zeros(total_points, dtype=float)
    kk = np.zeros(total_points, dtype=float)
    tt = np.zeros(total_points, dtype=float)
    pa_used = np.zeros(total_points, dtype=float)
    pb_used = np.zeros(total_points, dtype=float)
    point_indices = np.zeros((total_points, 2), dtype=np.int64)

    started = time.perf_counter()
    for out_index, point in enumerate(iter_scan_points(ctx, max_points=max_points, scan_indices=scan_indices)):
        matrix = assemble_hh_matrix(ctx, point, quadrature_order=quadrature_order)
        eigenvalues = sort_by_growth(solve_eigenvalues(matrix, ctx))
        ww[out_index, :] = eigenvalues[: ctx.nw]
        kxx[out_index] = point.kx
        kzz[out_index] = point.kz
        kk[out_index] = point.k
        tt[out_index] = point.theta_deg
        pa_used[out_index] = point.pa_value
        pb_used[out_index] = point.pb_value
        point_indices[out_index] = (point.jpa, point.jpb)

    is_full_scan = total_points == len(ctx.pa) * len(ctx.pb)
    if is_full_scan and len(ctx.pb) == 1:
        ww_out = ww.reshape(len(ctx.pa), ctx.nw)
        kxx_out = kxx.reshape(len(ctx.pa), 1)
        kzz_out = kzz.reshape(len(ctx.pa), 1)
        kk_out = kk.reshape(len(ctx.pa), 1)
        tt_out = tt.reshape(len(ctx.pa), 1)
    else:
        ww_out = ww
        kxx_out = kxx
        kzz_out = kzz
        kk_out = kk
        tt_out = tt

    if scan_indices is None and len(ctx.pb) == 1 and total_points <= len(ctx.pa):
        pa_out = ctx.pa[:total_points].copy()
        pb_out = ctx.pb.copy()
    elif is_full_scan:
        pa_out = ctx.pa.copy()
        pb_out = ctx.pb.copy()
    else:
        pa_out = pa_used.copy()
        pb_out = pb_used.copy()

    return ScanResult(
        ww=ww_out,
        pa=pa_out,
        pb=pb_out,
        kxx=kxx_out,
        kzz=kzz_out,
        kk=kk_out,
        tt=tt_out,
        elapsed_seconds=time.perf_counter() - started,
        pa_used=pa_used,
        pb_used=pb_used,
        pa_full=ctx.pa.copy(),
        pb_full=ctx.pb.copy(),
        scan_indices=point_indices,
    )


def _scale_scan_value(value: float, ilog: int) -> float:
    return float(value if ilog == 0 else 10.0**value)


def _scan_point_limit(ctx: SolverContext, max_points: int | None) -> int:
    full_scan_points = len(ctx.pa) * len(ctx.pb)
    if max_points is None or max_points < 0:
        return full_scan_points
    return min(int(max_points), full_scan_points)


def _selected_scan_point_indices(
    ctx: SolverContext,
    *,
    max_points: int | None,
    scan_indices: Iterable[int] | None,
) -> list[int]:
    full_scan_points = len(ctx.pa) * len(ctx.pb)
    if scan_indices is None:
        return list(range(_scan_point_limit(ctx, max_points)))

    selected = [int(value) for value in scan_indices]
    if not selected:
        raise ValueError("scan_indices must contain at least one flattened scan index")
    invalid = [value for value in selected if value < 0 or value >= full_scan_points]
    if invalid:
        raise ValueError(f"scan_indices out of range for {full_scan_points} points: {invalid}")
    if len(set(selected)) != len(selected):
        raise ValueError(f"scan_indices contains duplicates: {selected}")
    return selected

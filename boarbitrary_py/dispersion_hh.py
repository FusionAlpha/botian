from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erfc

from .coeffs import fun_in, precompute_abc
from .config import BOInput, FVCoefficients
from .io import load_fv_coefficients, read_bo_in
from .root_solvers import (
    ComplexRootSearchConfig,
    DEFAULT_PURE_GROWTH_SEARCH,
    ROOT_BRANCH_BOUNDARY,
    ROOT_BRANCH_COMPLEX_LM,
    ROOT_BRANCH_INTERIOR,
    ROOT_BRANCH_UNCONFIRMED,
    ROOT_METHOD_MATLAB_COMPLEX_LM,
    ROOT_METHOD_PURE_GROWTH,
    PureGrowthBranchResult,
    PureGrowthSearchConfig,
    solve_complex_lm_root,
    solve_pure_growth_branch,
)
from .zfuncs import fun_z_l


HH_ROOT_SUCCESS_RESIDUAL = DEFAULT_PURE_GROWTH_SEARCH.success_residual
HH_ROOT_BRANCH_INTERIOR = ROOT_BRANCH_INTERIOR
HH_ROOT_BRANCH_BOUNDARY = ROOT_BRANCH_BOUNDARY
HH_ROOT_BRANCH_UNCONFIRMED = ROOT_BRANCH_UNCONFIRMED
HH_ROOT_Y_MIN = DEFAULT_PURE_GROWTH_SEARCH.y_min
HH_ROOT_Y_MAX = DEFAULT_PURE_GROWTH_SEARCH.y_max
HH_ROOT_BOUNDARY_ATOL = DEFAULT_PURE_GROWTH_SEARCH.boundary_atol


@dataclass(frozen=True)
class HHRootPaths:
    root: Path
    matlab_root: Path
    genfv_dir: Path
    bo_in: Path


@dataclass(frozen=True)
class HHRootContext:
    paths: HHRootPaths
    bo_input: BOInput
    s_count: int
    n_harmonics: int
    b0: float
    theta_rad: float
    kk: np.ndarray
    c2: float
    epsilon0: float
    mu0: float
    kb: float
    qe: float
    mp: float
    me: float
    munit: float
    qs0: np.ndarray
    ms0: np.ndarray
    ns0: np.ndarray
    tzs0: np.ndarray
    tps0: np.ndarray
    vdsz0: np.ndarray
    vdsr0: np.ndarray
    ifv0: np.ndarray
    qs: np.ndarray
    ms: np.ndarray
    tzs: np.ndarray
    tps: np.ndarray
    vdsz: np.ndarray
    vdsr: np.ndarray
    vtzs: np.ndarray
    vtps: np.ndarray
    wps2: np.ndarray
    wcs: np.ndarray
    rhocs: np.ndarray
    ds: np.ndarray
    cap_as: np.ndarray
    nss: np.ndarray
    lsmax: np.ndarray
    msmax: np.ndarray
    aslm: tuple[np.ndarray, ...]
    wn: float
    kn: float


@dataclass(frozen=True)
class HHRootScanPoint:
    index: int
    k_norm: float
    kx: float
    kz: float


@dataclass(frozen=True)
class HHRootTerm:
    species: int
    harmonic: int
    ans: np.ndarray
    bns: np.ndarray
    cns: np.ndarray
    quadrature_order: int
    quadrature_extent: float


@dataclass(frozen=True)
class HHHarmonicContribution:
    species: int
    harmonic: int
    zeta: complex
    susceptibility: np.ndarray


@dataclass(frozen=True)
class HHRootEvaluation:
    determinant: complex
    tensor: np.ndarray
    dielectric_tensor: np.ndarray
    harmonic_contributions: tuple[HHHarmonicContribution, ...]
    quadrature_order: int
    quadrature_extent: float
    residual: float
    tensor_norm_2: float
    singular_values: np.ndarray
    sigma_min: float
    normalized_sigma_min: float
    polarization: np.ndarray


@dataclass(frozen=True)
class HHRootPointResult:
    index: int
    k_norm: float
    root: complex
    residual: float
    determinant: complex
    success: bool
    status: int
    branch_status: str
    physical_branch_confirmed: bool
    numerical_success: bool
    root_method: str
    message: str
    function_evaluations: int
    elapsed_seconds: float


@dataclass(frozen=True)
class HHRootScanResult:
    roots: np.ndarray
    kk: np.ndarray
    residuals: np.ndarray
    success: np.ndarray
    status: np.ndarray
    branch_status: tuple[str, ...]
    root_method: tuple[str, ...]
    physical_branch_confirmed: np.ndarray
    numerical_success: np.ndarray
    function_evaluations: np.ndarray
    elapsed_seconds: float
    point_results: tuple[HHRootPointResult, ...]


def default_hh_root_paths(root: str | Path | None = None) -> HHRootPaths:
    workspace = Path(root) if root is not None else Path.cwd()
    archive_root = workspace / "boarbitrary-main"
    matlab_root = archive_root / "em3dHH" / "boem3dHHroot"
    # The archived HH root script points at ../241221/genfv, which is not
    # present in this snapshot. The matching coefficient file is bundled with
    # the HH matrix case.
    genfv_dir = archive_root / "em3dHH" / "boHH" / "genfv"
    return HHRootPaths(root=archive_root, matlab_root=matlab_root, genfv_dir=genfv_dir, bo_in=matlab_root / "bo.in")


def load_hh_root_context(
    root: str | Path | None = None,
    *,
    n_harmonics: int = 2,
    b0: float = 0.1,
    theta_deg: float = 45.0,
    kk: np.ndarray | None = None,
) -> HHRootContext:
    paths = default_hh_root_paths(root)
    bo_input = read_bo_in(paths.bo_in)

    c2 = (2.99792458e8) ** 2
    epsilon0 = 8.854187817e-12
    mu0 = 1.0 / (c2 * epsilon0)
    kb = 1.38064852e-23
    qe = 1.60217662e-19
    mp = 1.6726219e-27
    me = 9.1094e-31
    munit = mp

    species = bo_input.species
    s_count = len(species)
    qs0 = np.array([sp.charge_qe for sp in species], dtype=float)
    ms0 = np.array([sp.mass_munit for sp in species], dtype=float)
    ns0 = np.array([sp.density for sp in species], dtype=float)
    tzs0 = np.array([sp.tz_ev for sp in species], dtype=float)
    tps0 = np.array([sp.tp_ev for sp in species], dtype=float)
    vdsz0 = np.array([sp.drift_z_c for sp in species], dtype=float)
    vdsr0 = np.array([sp.drift_r_c for sp in species], dtype=float)
    ifv0 = np.array([sp.ifv for sp in species], dtype=int)

    qs = qs0 * qe
    ms = ms0 * munit

    aslm: list[np.ndarray] = []
    fv_coefficients: dict[int, FVCoefficients] = {}
    lsmax = np.zeros(s_count, dtype=int)
    msmax = np.zeros(s_count, dtype=int)

    # First use input temperatures for analytic species. Grid species are
    # overwritten by the coefficient metadata exactly like the MATLAB root path.
    vdsz = vdsz0 * np.sqrt(c2)
    vdsr = vdsr0 * np.sqrt(c2)
    for idx, species_input in enumerate(species, start=1):
        if species_input.ifv == 1:
            aslm.append(np.ones((1, 1), dtype=np.complex128))
            continue
        coeffs = load_fv_coefficients(paths.genfv_dir / f"fvceff{idx}.mat")
        fv_coefficients[idx] = coeffs
        aslm.append(np.asarray(coeffs.alm, dtype=np.complex128))
        vdsz[idx - 1] = coeffs.vdz
        vdsr[idx - 1] = coeffs.vdr
        vdsz0[idx - 1] = coeffs.vdz / np.sqrt(c2)
        vdsr0[idx - 1] = coeffs.vdr / np.sqrt(c2)
        tzs0[idx - 1] = ms[idx - 1] * coeffs.vtz**2 / (2 * qe)
        tps0[idx - 1] = ms[idx - 1] * coeffs.vtp**2 / (2 * qe)

    tzs = tzs0 * qe / kb
    tps = tps0 * qe / kb
    vtzs = np.sqrt(2 * kb * tzs / ms)
    vtps = np.sqrt(2 * kb * tps / ms)
    wps2 = ns0 * qs**2 / ms / epsilon0
    wcs = b0 * qs / ms
    rhocs = np.sqrt(kb * tps / ms) / wcs
    ds = vdsr / vtps
    cap_as = np.exp(-(ds**2)) + np.sqrt(np.pi) * ds * erfc(-ds)

    for idx, alm in enumerate(aslm):
        lsmax[idx] = alm.shape[0] - 1
        msmax[idx] = alm.shape[1] - 1

    ns = np.full(s_count, 2 * n_harmonics + 1, dtype=int)
    nss = np.floor((ns - 1) / 2).astype(int)
    wn = abs(wcs[0])
    va = b0 / np.sqrt(mu0 * np.sum(ms * ns0))
    kn = wn / va

    if kk is None:
        kk = np.arange(0.1, 0.4000001, 0.025, dtype=float)

    return HHRootContext(
        paths=paths,
        bo_input=bo_input,
        s_count=s_count,
        n_harmonics=n_harmonics,
        b0=b0,
        theta_rad=np.deg2rad(theta_deg),
        kk=np.asarray(kk, dtype=float),
        c2=c2,
        epsilon0=epsilon0,
        mu0=mu0,
        kb=kb,
        qe=qe,
        mp=mp,
        me=me,
        munit=munit,
        qs0=qs0,
        ms0=ms0,
        ns0=ns0,
        tzs0=tzs0,
        tps0=tps0,
        vdsz0=vdsz0,
        vdsr0=vdsr0,
        ifv0=ifv0,
        qs=qs,
        ms=ms,
        tzs=tzs,
        tps=tps,
        vdsz=vdsz,
        vdsr=vdsr,
        vtzs=vtzs,
        vtps=vtps,
        wps2=wps2,
        wcs=wcs,
        rhocs=rhocs,
        ds=ds,
        cap_as=cap_as,
        nss=nss,
        lsmax=lsmax,
        msmax=msmax,
        aslm=tuple(aslm),
        wn=float(wn),
        kn=float(kn),
    )


def iter_hh_root_scan_points(ctx: HHRootContext, *, max_points: int | None = None):
    limit = len(ctx.kk) if max_points is None or max_points <= 0 else min(max_points, len(ctx.kk))
    for index, k_norm in enumerate(ctx.kk[:limit]):
        yield HHRootScanPoint(
            index=index,
            k_norm=float(k_norm),
            kx=float(k_norm * ctx.kn * np.sin(ctx.theta_rad)),
            kz=float(k_norm * ctx.kn * np.cos(ctx.theta_rad)),
        )


def precompute_hh_root_terms(
    ctx: HHRootContext,
    point: HHRootScanPoint,
    *,
    quadrature_order: int = 120,
    quadrature_extent: float = 10.0,
) -> tuple[HHRootTerm, ...]:
    kx = point.kx if point.kx != 0 else 1e-30
    a_s = kx * ctx.rhocs * np.sqrt(2.0)
    terms: list[HHRootTerm] = []
    for s in range(ctx.s_count):
        for n in range(-int(ctx.nss[s]), int(ctx.nss[s]) + 1):
            ans, bns, cns = precompute_abc(
                n,
                float(a_s[s]),
                float(ctx.ds[s]),
                int(ctx.msmax[s]),
                float(ctx.cap_as[s]),
                order=quadrature_order,
                extent=quadrature_extent,
            )
            terms.append(
                HHRootTerm(
                    species=s,
                    harmonic=n,
                    ans=ans,
                    bns=bns,
                    cns=cns,
                    quadrature_order=int(quadrature_order),
                    quadrature_extent=float(quadrature_extent),
                )
            )
    return tuple(terms)


def evaluate_hh_root_determinant(
    ctx: HHRootContext,
    point: HHRootScanPoint,
    w: complex,
    terms: tuple[HHRootTerm, ...] | None = None,
    *,
    quadrature_order: int = 120,
    quadrature_extent: float = 10.0,
) -> HHRootEvaluation:
    kx = point.kx if point.kx != 0 else 1e-30
    kz = point.kz if point.kz != 0 else 1.0e-4 * kx
    w = complex(w)
    if terms is None:
        terms = precompute_hh_root_terms(
            ctx,
            point,
            quadrature_order=quadrature_order,
            quadrature_extent=quadrature_extent,
        )
    actual_quadrature_order = terms[0].quadrature_order if terms else int(quadrature_order)
    actual_quadrature_extent = terms[0].quadrature_extent if terms else float(quadrature_extent)

    nx = np.sqrt(ctx.c2) * kx / w
    nz = np.sqrt(ctx.c2) * kz / w
    k_tensor = np.eye(3, dtype=np.complex128)
    harmonic_contributions: list[HHHarmonicContribution] = []

    for term in terms:
        s = term.species
        n = term.harmonic
        ans, bns, cns = term.ans, term.bns, term.cns
        zetasn = (w - kz * ctx.vdsz[s] - n * ctx.wcs[s]) / (kz * ctx.vtzs[s])
        vdz_over_vtz = ctx.vdsz[s] / ctx.vtzs[s]
        vtps_over_vtzs = ctx.vtps[s] / ctx.vtzs[s]
        aslm = ctx.aslm[s]

        x_xx = 0j
        x_xy = 0j
        x_xz = 0j
        x_yy = 0j
        x_yz = 0j
        x_zz = 0j
        for ell in range(int(ctx.lsmax[s]) + 1):
            z_l = fun_z_l(zetasn, ell)
            z_lm1 = fun_z_l(zetasn, ell - 1)
            z_lp1 = fun_z_l(zetasn, ell + 1)
            z_lp2 = fun_z_l(zetasn, ell + 2)
            z_lp3 = fun_z_l(zetasn, ell + 3)
            i_l = fun_in(ell)
            i_lm1 = fun_in(ell - 1)
            i_lp1 = fun_in(ell + 1)
            i_lp2 = fun_in(ell + 2)

            z1 = 2 * z_lp1 - ell * z_lm1
            z2 = 2 * z_lp2 - ell * z_l
            for m in range(int(ctx.msmax[s]) + 1):
                coef = aslm[ell, m]
                radial_a = 2 * ans[m + 2, 0] - m * ans[m, 0]
                radial_b = 2 * bns[m + 2, 0] - m * bns[m, 0]
                radial_c = 2 * cns[m + 2, 0] - m * cns[m, 0]
                a_m = ans[m + 1, 1]
                b_m = bns[m + 1, 1]
                c_m = cns[m + 1, 1]

                cyclotron_z = n * ctx.wcs[s] / (kz * ctx.vtzs[s])
                common = cyclotron_z * z_l - i_l
                x_xx += coef * (common * radial_a + a_m * vtps_over_vtzs**2 * z1)
                x_xy += coef * (common * radial_b + b_m * vtps_over_vtzs**2 * z1)
                x_yy += coef * (common * radial_c + c_m * vtps_over_vtzs**2 * z1)
                x_xz += coef * (
                    n
                    * ctx.wcs[s]
                    / (kz * ctx.vtps[s])
                    * (vdz_over_vtz * z_l + z_lp1)
                    * radial_a
                    + a_m * vtps_over_vtzs * (z2 + vdz_over_vtz * z1)
                )
                x_yz += coef * (
                    n
                    * ctx.wcs[s]
                    / (kz * ctx.vtps[s])
                    * (vdz_over_vtz * z_l + z_lp1)
                    * radial_b
                    + b_m * vtps_over_vtzs * (z2 + vdz_over_vtz * z1)
                )
                x_zz += coef * (
                    n
                    * ctx.wcs[s]
                    / (kz * ctx.vtzs[s])
                    * (vdz_over_vtz**2 * z_l + 2 * vdz_over_vtz * z_lp1 + z_lp2)
                    * radial_a
                    + a_m
                    * vtps_over_vtzs**2
                    * (
                        (
                            2 * vdz_over_vtz * (z_lp2 - i_lp1)
                            + 2 * (z_lp3 - i_lp2)
                            - ell * vdz_over_vtz * (z_l - i_lm1)
                            - ell * (z_lp1 - i_l)
                        )
                        + vdz_over_vtz * (vdz_over_vtz * z1 + 2 * (z_lp2 - ell * z_l))
                    )
                )

        x_xx *= (n * ctx.wcs[s] / (kx * ctx.vtps[s])) ** 2
        x_xy *= 1j * (n * ctx.wcs[s] / (kx * ctx.vtps[s]))
        x_xz *= n * ctx.wcs[s] / (kx * ctx.vtps[s])
        x_yz *= -1j
        x_zz *= (ctx.vtzs[s] / ctx.vtps[s]) ** 2

        x_yx = -x_xy
        x_zx = x_xz
        x_zy = -x_yz
        term1 = ctx.wps2[s] / w**2
        susceptibility = term1 * np.array(
            [
                [x_xx, x_xy, x_xz],
                [x_yx, x_yy, x_yz],
                [x_zx, x_zy, x_zz],
            ],
            dtype=np.complex128,
        )
        k_tensor += susceptibility
        harmonic_contributions.append(
            HHHarmonicContribution(
                species=s,
                harmonic=n,
                zeta=complex(zetasn),
                susceptibility=susceptibility,
            )
        )

    d_tensor = np.array(
        [
            [k_tensor[0, 0] - nz**2, k_tensor[0, 1], k_tensor[0, 2] + nx * nz],
            [k_tensor[1, 0], k_tensor[1, 1] - (nx**2 + nz**2), k_tensor[1, 2]],
            [k_tensor[2, 0] + nx * nz, k_tensor[2, 1], k_tensor[2, 2] - nx**2],
        ],
        dtype=np.complex128,
    )
    determinant = complex(np.linalg.det(d_tensor))
    scale = max(1.0, float(np.max(np.abs(d_tensor))) ** 3)
    _, singular_values, vh = np.linalg.svd(d_tensor)
    tensor_norm_2 = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    polarization = _canonicalize_mode(vh.conj().T[:, -1])
    return HHRootEvaluation(
        determinant=determinant,
        tensor=d_tensor,
        dielectric_tensor=k_tensor,
        harmonic_contributions=tuple(harmonic_contributions),
        quadrature_order=actual_quadrature_order,
        quadrature_extent=actual_quadrature_extent,
        residual=float(abs(determinant) / scale),
        tensor_norm_2=tensor_norm_2,
        singular_values=np.asarray(singular_values, dtype=float),
        sigma_min=sigma_min,
        normalized_sigma_min=float(sigma_min / (1.0 + tensor_norm_2)),
        polarization=polarization,
    )


def _canonicalize_mode(mode: np.ndarray) -> np.ndarray:
    mode = np.asarray(mode, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(mode))
    if norm == 0.0 or not np.isfinite(norm):
        return mode
    mode = mode / norm
    pivot = int(np.argmax(np.abs(mode)))
    if abs(mode[pivot]) > 0.0:
        mode = mode * np.exp(-1j * np.angle(mode[pivot]))
    return mode


def solve_hh_root_point(
    ctx: HHRootContext,
    point: HHRootScanPoint,
    *,
    initial_guess: complex | None = None,
    quadrature_order: int = 120,
    xtol: float = 1e-11,
    root_method: str = ROOT_METHOD_PURE_GROWTH,
) -> HHRootPointResult:
    terms = precompute_hh_root_terms(ctx, point, quadrature_order=quadrature_order)
    guess = initial_guess if initial_guess is not None else 0.01j * ctx.wn

    def residual_vector(values: np.ndarray) -> np.ndarray:
        w = complex(values[0], values[1]) * ctx.wn
        evaluation = evaluate_hh_root_determinant(ctx, point, w, terms)
        scale = max(1.0, float(np.max(np.abs(evaluation.tensor))) ** 3)
        normalized = evaluation.determinant / scale
        return np.array([normalized.real, normalized.imag], dtype=float)

    def residual_norm_squared(values: np.ndarray) -> float:
        vector = residual_vector(values)
        return float(vector @ vector)

    start = time.perf_counter()
    normalized_guess = np.array([guess.real / ctx.wn, guess.imag / ctx.wn], dtype=float)
    if root_method == ROOT_METHOD_PURE_GROWTH:
        branch_result = _solve_pure_growth_branch(
            residual_norm_squared,
            normalized_guess=normalized_guess,
            xtol=xtol,
        )
        branch_root = branch_result.normalized_root
        branch_status = branch_result.branch_status
        status = branch_result.status
        function_evaluations = branch_result.function_evaluations
        message = branch_result.message
        branch_numerical_success = branch_result.numerical_success
        branch_physical_confirmed = branch_result.physical_branch_confirmed
    elif root_method == ROOT_METHOD_MATLAB_COMPLEX_LM:
        complex_result = solve_complex_lm_root(
            residual_vector,
            normalized_guess=normalized_guess,
            config=ComplexRootSearchConfig(success_residual=HH_ROOT_SUCCESS_RESIDUAL, xtol=xtol, ftol=xtol, gtol=xtol),
        )
        branch_root = complex_result.normalized_root
        branch_status = ROOT_BRANCH_COMPLEX_LM
        status = complex_result.status
        function_evaluations = complex_result.function_evaluations
        message = complex_result.message
        branch_numerical_success = complex_result.numerical_success
        branch_physical_confirmed = False
    else:
        raise ValueError(f"Unsupported root_method={root_method!r}")

    elapsed = time.perf_counter() - start
    solved_w = complex(branch_root[0], branch_root[1]) * ctx.wn
    evaluation = evaluate_hh_root_determinant(ctx, point, solved_w, terms)
    numerical_success = bool(branch_numerical_success or evaluation.residual <= HH_ROOT_SUCCESS_RESIDUAL)
    physical_branch_confirmed = bool(
        root_method == ROOT_METHOD_PURE_GROWTH
        and branch_status == HH_ROOT_BRANCH_INTERIOR
        and branch_physical_confirmed
        and evaluation.residual <= HH_ROOT_SUCCESS_RESIDUAL
    )
    success = physical_branch_confirmed if root_method == ROOT_METHOD_PURE_GROWTH else numerical_success
    return HHRootPointResult(
        index=point.index,
        k_norm=point.k_norm,
        root=solved_w,
        residual=evaluation.residual,
        determinant=evaluation.determinant,
        success=success,
        status=status,
        branch_status=branch_status,
        physical_branch_confirmed=physical_branch_confirmed,
        numerical_success=numerical_success,
        root_method=root_method,
        message=message,
        function_evaluations=function_evaluations,
        elapsed_seconds=float(elapsed),
    )


def _solve_pure_growth_branch(
    residual_norm_squared,
    *,
    normalized_guess: np.ndarray,
    xtol: float,
    config: PureGrowthSearchConfig = DEFAULT_PURE_GROWTH_SEARCH,
) -> PureGrowthBranchResult:
    return solve_pure_growth_branch(
        residual_norm_squared,
        normalized_guess=normalized_guess,
        xtol=xtol,
        config=config,
    )


def run_hh_root_scan(
    ctx: HHRootContext | None = None,
    *,
    root_path: str | Path | None = None,
    max_points: int | None = 1,
    quadrature_order: int = 120,
    continuation: bool = False,
    root_method: str = ROOT_METHOD_PURE_GROWTH,
) -> HHRootScanResult:
    ctx = ctx or load_hh_root_context(root_path)
    point_results: list[HHRootPointResult] = []
    guess: complex | None = None
    start = time.perf_counter()
    for point in iter_hh_root_scan_points(ctx, max_points=max_points):
        point_result = solve_hh_root_point(
            ctx,
            point,
            initial_guess=guess,
            quadrature_order=quadrature_order,
            root_method=root_method,
        )
        point_results.append(point_result)
        if continuation:
            guess = point_result.root
    elapsed = time.perf_counter() - start

    roots = np.array([item.root for item in point_results], dtype=np.complex128)
    kk = np.array([item.k_norm for item in point_results], dtype=float)
    residuals = np.array([item.residual for item in point_results], dtype=float)
    success = np.array([item.success for item in point_results], dtype=bool)
    status = np.array([item.status for item in point_results], dtype=int)
    branch_status = tuple(item.branch_status for item in point_results)
    root_methods = tuple(item.root_method for item in point_results)
    physical_branch_confirmed = np.array([item.physical_branch_confirmed for item in point_results], dtype=bool)
    numerical_success = np.array([item.numerical_success for item in point_results], dtype=bool)
    function_evaluations = np.array([item.function_evaluations for item in point_results], dtype=int)
    return HHRootScanResult(
        roots=roots,
        kk=kk,
        residuals=residuals,
        success=success,
        status=status,
        branch_status=branch_status,
        root_method=root_methods,
        physical_branch_confirmed=physical_branch_confirmed,
        numerical_success=numerical_success,
        function_evaluations=function_evaluations,
        elapsed_seconds=float(elapsed),
        point_results=tuple(point_results),
    )

from __future__ import annotations

from functools import lru_cache
from math import sqrt

import numpy as np
from scipy.special import ive

from ..coeffs import fun_an, fun_bn, fun_cn
from .config import LegacyCoefficients, LegacyContext, LegacyScanPoint


@lru_cache(maxsize=32768)
def legacy_fun_an(n: int, a: float, b: float, c: float, order: int = 120) -> float:
    """Evaluate the legacy BO ``funAn(n,a,b,c)`` ring integral."""

    if c == b:
        return fun_an(n, a, b, 1, 0, order)
    if c == 0.0:
        return fun_an(n, a, b, 0, 1, order)
    return fun_an(n, a, b, 0, 1, order) - c * fun_an(n, a, b, 0, 0, order)


@lru_cache(maxsize=32768)
def legacy_fun_bn(n: int, a: float, b: float, c: float, order: int = 120) -> float:
    """Evaluate the legacy BO ``funBn(n,a,b,c)`` ring integral."""

    if c == b:
        return fun_bn(n, a, b, 1, 1, order)
    if c == 0.0:
        return fun_bn(n, a, b, 0, 2, order)
    return fun_bn(n, a, b, 0, 2, order) - c * fun_bn(n, a, b, 0, 1, order)


@lru_cache(maxsize=32768)
def legacy_fun_cn(n: int, a: float, b: float, c: float, order: int = 120) -> float:
    """Evaluate the legacy BO ``funCn(n,a,b,c)`` ring integral."""

    if c == b:
        return fun_cn(n, a, b, 1, 2, order)
    if c == 0.0:
        return fun_cn(n, a, b, 0, 3, order)
    return fun_cn(n, a, b, 0, 3, order) - c * fun_cn(n, a, b, 0, 2, order)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    result.setflags(write=False)
    return result


def _es_magnetized_a_terms(
    ctx: LegacyContext,
    *,
    species: int,
    harmonic: int,
    a_values: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(ctx.nrsab[species])
    anabb = np.empty(count, dtype=float)
    anab0 = np.empty(count, dtype=float)
    for iab in range(count):
        if ctx.vdsr[species] == 0.0:
            as2 = 0.5 * a_values[iab] ** 2
            gamma_n = float(ive(harmonic, as2))
            value = 2.0 * gamma_n / ctx.asab_norm[iab, species]
            anabb[iab] = value
            anab0[iab] = value
        else:
            b = float(ctx.bsab[iab, species])
            scale = 4.0 / ctx.asab_norm[iab, species]
            a = float(a_values[iab])
            anabb[iab] = scale * legacy_fun_an(harmonic, a, b, b, quadrature_order)
            anab0[iab] = scale * legacy_fun_an(harmonic, a, b, 0.0, quadrature_order)
    return anabb, anab0


def evaluate_es3d_coefficients(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    quadrature_order: int = 120,
) -> LegacyCoefficients:
    """Build the ordered ES3D pole and residue arrays used by BO v190307."""

    if ctx.case.model.iem != 0:
        raise ValueError(f"ES3D coefficients require iem=0, got iem={ctx.case.model.iem}")
    if point.k == 0.0:
        raise ValueError("The legacy ES3D matrix is undefined at k=0")

    a_values = point.kx * ctx.rhocsab * sqrt(2.0)
    kvtsab = np.sqrt(
        (point.kx * ctx.vtpsab) ** 2
        + (point.kz * ctx.vtzs[None, :]) ** 2
    )
    cz_signed = -ctx.czj if point.kz < 0.0 else ctx.czj
    poles = np.empty(ctx.snj, dtype=np.complex128)
    residues = np.zeros(ctx.snj, dtype=np.complex128)
    a_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for state in range(ctx.snj):
        species = int(ctx.index_map.species[state])
        harmonic_or_sigma = int(ctx.index_map.harmonic_or_sigma[state])
        pole_index = int(ctx.index_map.pole[state])
        if ctx.index_map.unmagnetized[state]:
            iab = harmonic_or_sigma - 1
            kvt = kvtsab[iab, species]
            if kvt == 0.0:
                raise ValueError("The legacy unmagnetized coefficient is undefined for k*v_t=0")
            poles[state] = (
                ctx.czj[pole_index] * kvt
                + point.kz * ctx.vdsz[species]
                + point.kx * ctx.vdsx[species]
                - 1j * ctx.nus[species]
            )
            residues[state] = (
                2.0
                * ctx.rsab[iab, species]
                * ctx.wps2[species]
                * ctx.bzj[pole_index]
                * ctx.czj[pole_index]
                / kvt
            )
            continue

        harmonic = harmonic_or_sigma
        poles[state] = (
            cz_signed[pole_index] * point.kz * ctx.vtzs[species]
            + point.kz * ctx.vdsz[species]
            + point.kx * ctx.vdsx[species]
            - 1j * ctx.nus[species]
            + harmonic * ctx.wcs[species]
        )
        key = (species, harmonic)
        if key not in a_cache:
            a_cache[key] = _es_magnetized_a_terms(
                ctx,
                species=species,
                harmonic=harmonic,
                a_values=a_values[:, species],
                quadrature_order=quadrature_order,
            )
        anabb, anab0 = a_cache[key]
        for iab in range(int(ctx.nrsab[species])):
            residues[state] += (
                ctx.rsab[iab, species]
                * ctx.wps2[species]
                * ctx.bzj[pole_index]
                * (
                    anab0[iab]
                    * point.kz
                    / ctx.vtzs[species]
                    * cz_signed[pole_index]
                    + anabb[iab]
                    * harmonic
                    * ctx.wcs[species]
                    / ctx.vtpsab[iab, species] ** 2
                )
                / point.k**2
            )

    return LegacyCoefficients(poles=_readonly(poles), es_residues=_readonly(residues))


def effective_em_wavevector(point: LegacyScanPoint) -> tuple[float, float, float]:
    """Return the legacy EM wavevector after its exact parallel-edge convention."""

    kx = 1e-30 if point.kx == 0.0 else point.kx
    kz = point.kz
    return float(np.hypot(kx, kz)), float(kx), float(kz)


def _em_magnetized_abc_terms(
    ctx: LegacyContext,
    *,
    species: int,
    harmonic: int,
    a_values: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = int(ctx.nrsab[species])
    anabb = np.empty(count, dtype=float)
    anab0 = np.empty(count, dtype=float)
    bnabb = np.empty(count, dtype=float)
    bnab0 = np.empty(count, dtype=float)
    cnabb = np.empty(count, dtype=float)
    cnab0 = np.empty(count, dtype=float)
    for iab in range(count):
        normalization = ctx.asab_norm[iab, species]
        a = float(a_values[iab])
        if ctx.vdsr[species] == 0.0:
            as2 = 0.5 * a**2
            gamma_n = float(ive(harmonic, as2))
            gamma_np = float(
                (
                    ive(harmonic + 1, as2)
                    + ive(harmonic - 1, as2)
                    - 2.0 * gamma_n
                )
                / 2.0
            )
            anabb[iab] = 2.0 * gamma_n / normalization
            anab0[iab] = anabb[iab]
            bnabb[iab] = gamma_np * a / normalization
            bnab0[iab] = bnabb[iab]
            cnabb[iab] = (
                gamma_n * harmonic**2 / as2 - 2.0 * gamma_np * as2
            ) / normalization
            cnab0[iab] = cnabb[iab]
        else:
            b = float(ctx.bsab[iab, species])
            scale = 4.0 / normalization
            anabb[iab] = scale * legacy_fun_an(harmonic, a, b, b, quadrature_order)
            anab0[iab] = scale * legacy_fun_an(harmonic, a, b, 0.0, quadrature_order)
            bnabb[iab] = scale * legacy_fun_bn(harmonic, a, b, b, quadrature_order)
            bnab0[iab] = scale * legacy_fun_bn(harmonic, a, b, 0.0, quadrature_order)
            cnabb[iab] = scale * legacy_fun_cn(harmonic, a, b, b, quadrature_order)
            cnab0[iab] = scale * legacy_fun_cn(harmonic, a, b, 0.0, quadrature_order)
    return anabb, anab0, bnabb, bnab0, cnabb, cnab0


def _em_unmagnetized_residue(
    ctx: LegacyContext,
    *,
    species: int,
    sigma: int,
    pole_index: int,
    pole: complex,
    k: float,
    kx: float,
    kz: float,
    kvtsab: np.ndarray,
) -> np.ndarray:
    iab = sigma - 1
    c = ctx.czj[pole_index]
    vtp = ctx.vtpsab[iab, species]
    vtp2 = vtp**2
    vtz = ctx.vtzs[species]
    vtz2 = vtz**2
    kvt = kvtsab[iab, species]
    kvt2 = kvt**2
    vx = ctx.vdsx[species]
    vy = ctx.vdsy[species]
    vz = ctx.vdsz[species]
    nu = ctx.nus[species]
    anisotropy = ctx.lmd_tab[iab, species] - 1.0
    k2 = k**2

    p11 = (
        -1j
        * nu
        * ((kx * vtp) ** 2 * c**2 + kx * vx * kvt * c + 0.5 * (kz * vtz) ** 2)
        / kvt2
        + (k * kx * vtp2) ** 2 / kvt**3 * c**3
        + 2.0 * k2 * kx * vtp2 / kvt2 * vx * c**2
        + (
            0.5 * vtp2 * k2 * kz**2 * vtz2 / kvt**3
            + k2 * vx**2 / kvt
            + anisotropy * kx**2 * kz**2 * vtp2**2 / kvt**3
        )
        * c
        + anisotropy * kx * kz**2 / kvt2 * vtp2 * vx
    )
    p12 = (
        vy
        * (
            k2 * kx * vtp2 * c**2
            + k2 * vx * kvt * c
            + 0.5 * anisotropy * kx * kz**2 * vtp2
        )
        / kvt2
    )
    p21 = (
        vy
        * (
            -1j * nu * kx * kvt * c
            + k2 * kx * vtp2 * c**2
            + k2 * vx * kvt * c
            + 0.5 * anisotropy * kx * kz**2 * vtp2
        )
        / kvt2
    )
    p22 = -0.5j * nu + (0.5 * vtp2 + vy**2) * k2 / kvt * c
    common_cubic = k2 * kx * kz * vtz2 * vtp2 / kvt**3 * c**3
    common_square = k2 * (kx * vtp2 * vz + kz * vtz2 * vx) / kvt2 * c**2
    common_linear = (
        k2 * vx * vz * kvt2
        - 0.5 * k2 * kx * kz * vtz2 * vtp2
        - 0.5
        * anisotropy
        * (kx**2 * vtp2 - kz**2 * vtz2)
        * kx
        * kz
        * vtp2
    ) / kvt**3 * c
    common_constant = (
        0.5 * anisotropy * kx * kz * vtp2 * (kz * vz - kx * vx) / kvt2
    )
    p13 = (
        -1j
        * nu
        * kz
        * (kx * vtp2 * c**2 + vx * kvt * c - 0.5 * kx * vtp2)
        / kvt2
        + common_cubic
        + common_square
        + common_linear
        + common_constant
    )
    p31 = (
        -1j
        * nu
        * kx
        * (kz * vtz2 * c**2 + vz * kvt * c - 0.5 * kz * vtz2)
        / kvt2
        + common_cubic
        + common_square
        + common_linear
        + common_constant
    )
    p23 = vy * (
        -1j * nu * kz / kvt * c
        + (
            k2 * kz * vtz2 * c**2
            + k2 * vz * kvt * c
            - 0.5 * anisotropy * kx**2 * kz * vtp2
        )
        / kvt2
    )
    p32 = (
        vy
        * (
            k2 * kz * vtz2 * c**2
            + k2 * vz * kvt * c
            - 0.5 * anisotropy * kx**2 * kz * vtp2
        )
        / kvt2
    )
    p33 = (
        -1j
        * nu
        * (kz**2 * vtz2 * c**2 + kz * vz * kvt * c + 0.5 * kx**2 * vtp2)
        / kvt2
        + k2 * kz**2 * vtz2**2 / kvt**3 * c**3
        + 2.0 * k2 * kz * vtz2 * vz / kvt2 * c**2
        + (
            k2 * vz**2 * kvt2
            + 0.5 * k2 * kx**2 * vtz2 * vtp2
            - anisotropy * kx**2 * kz**2 * vtz2 * vtp2
        )
        / kvt**3
        * c
        - anisotropy * kx**2 * kz * vtp2 * vz / kvt2
    )
    p = np.asarray(
        ((p11, p12, p13), (p21, p22, p23), (p31, p32, p33)),
        dtype=np.complex128,
    )
    scale = (
        ctx.rsab[iab, species]
        * ctx.wps2[species]
        * 2.0
        * ctx.bzj[pole_index]
        / pole
    )
    return scale * p


def _em_magnetized_residue(
    ctx: LegacyContext,
    *,
    species: int,
    harmonic: int,
    pole_index: int,
    pole: complex,
    kx: float,
    kz: float,
    abc: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cz_signed: np.ndarray,
) -> np.ndarray:
    anabb, anab0, bnabb, bnab0, cnabb, cnab0 = abc
    residue = np.zeros((3, 3), dtype=np.complex128)
    nw = harmonic * ctx.wcs[species]
    vx = ctx.vdsx[species]
    vy = ctx.vdsy[species]
    vz = ctx.vdsz[species]
    nu = ctx.nus[species]
    vtz = ctx.vtzs[species]
    cz = cz_signed[pole_index]
    for iab in range(int(ctx.nrsab[species])):
        vtp = ctx.vtpsab[iab, species]
        vtp2 = vtp**2
        a_x = nw / kx + vx
        p11 = a_x * (
            nw / kx * (nw + kx * vx - 1j * nu) * anabb[iab] / vtp2
            + anab0[iab] * a_x * kz * cz / vtz
        )
        p12 = a_x * (
            vy * (nw * anabb[iab] / vtp2 + anab0[iab] * kz * cz / vtz)
            + 1j
            * (
                (nw - 1j * nu) * bnabb[iab] / vtp
                + bnab0[iab] * vtp / vtz * kz * cz
            )
        )
        p21 = (
            nw
            / kx
            * (nw + kx * vx - 1j * nu)
            * (-1j * vtp * bnabb[iab] + vy * anabb[iab])
            / vtp2
            + a_x
            / vtz
            * (-1j * vtp * bnab0[iab] + vy * anab0[iab])
            * kz
            * cz
        )
        p22 = (
            vy
            * nw
            * (-1j * vtp * bnabb[iab] + vy * anabb[iab])
            / vtp2
            + vy
            * (-1j * vtp * bnab0[iab] + vy * anab0[iab])
            / vtz
            * kz
            * cz
            + (nw - 1j * nu) * (cnabb[iab] + 1j * vy * bnabb[iab] / vtp)
            + vtp
            / vtz
            * (vtp * cnab0[iab] + 1j * vy * bnab0[iab])
            * kz
            * cz
        )
        p13 = a_x * (
            nw * (cz * vtz + vz) * anabb[iab] / vtp2
            + anab0[iab] * (kz * cz**2 + (kz * vz - 1j * nu) / vtz * cz)
        )
        p31 = (
            nw
            / kx
            * (nw + kx * vx - 1j * nu)
            * (cz * vtz + vz)
            * anabb[iab]
            / vtp2
            + anab0[iab] * a_x * (cz + vz / vtz) * kz * cz
        )
        p23 = (
            (vy * anabb[iab] - 1j * vtp * bnabb[iab])
            * (cz * vtz + vz)
            * nw
            / vtp2
            + (vy * anab0[iab] - 1j * vtp * bnab0[iab])
            * (kz * cz + (kz * vz - 1j * nu) / vtz)
            * cz
        )
        p32 = (
            vy * (nw * anabb[iab] * cz * vtz / vtp2 + anab0[iab] * kz**2 * cz**2)
            + vy
            * vz
            * (nw * anabb[iab] / vtp2 + anab0[iab] * kz * cz / vtz)
            + 1j
            * (
                (nw - 1j * nu) * bnabb[iab] * vtz / vtp
                + bnab0[iab] * kz * vtp * cz
            )
            * (cz + vz / vtz)
        )
        p33 = (
            nw * (vtz**2 * cz + vtz * vz) * anabb[iab] / vtp2
            + anab0[iab] * cz * (cz * kz * vtz + (kz * vz - 1j * nu))
        ) * (cz + vz / vtz)
        p = np.asarray(
            ((p11, p12, p13), (p21, p22, p23), (p31, p32, p33)),
            dtype=np.complex128,
        )
        scale = (
            ctx.rsab[iab, species]
            * ctx.wps2[species]
            * ctx.bzj[pole_index]
            / pole
        )
        residue += scale * p
    return residue


def evaluate_em3d_coefficients(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    quadrature_order: int = 120,
) -> LegacyCoefficients:
    """Build ordered EM3D pole/residue tensors and the constant current block."""

    if ctx.case.model.iem not in {1, 2}:
        raise ValueError(f"EM3D coefficients require iem=1 or iem=2, got {ctx.case.model.iem}")
    k, kx, kz = effective_em_wavevector(point)
    if k == 0.0:
        raise ValueError("The legacy EM3D matrix is undefined at k=0")

    a_values = kx * ctx.rhocsab * sqrt(2.0)
    kvtsab = np.sqrt((kx * ctx.vtpsab) ** 2 + (kz * ctx.vtzs[None, :]) ** 2)
    cz_signed = -ctx.czj if kz < 0.0 else ctx.czj
    poles = np.empty(ctx.snj, dtype=np.complex128)
    residues = np.zeros((ctx.snj, 3, 3), dtype=np.complex128)
    abc_cache: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    for state in range(ctx.snj):
        species = int(ctx.index_map.species[state])
        harmonic_or_sigma = int(ctx.index_map.harmonic_or_sigma[state])
        pole_index = int(ctx.index_map.pole[state])
        if ctx.index_map.unmagnetized[state]:
            sigma = harmonic_or_sigma
            iab = sigma - 1
            kvt = kvtsab[iab, species]
            if kvt == 0.0:
                raise ValueError("The legacy unmagnetized coefficient is undefined for k*v_t=0")
            poles[state] = (
                ctx.czj[pole_index] * kvt
                + kz * ctx.vdsz[species]
                + kx * ctx.vdsx[species]
                - 1j * ctx.nus[species]
            )
            residues[state] = _em_unmagnetized_residue(
                ctx,
                species=species,
                sigma=sigma,
                pole_index=pole_index,
                pole=poles[state],
                k=k,
                kx=kx,
                kz=kz,
                kvtsab=kvtsab,
            )
            continue

        harmonic = harmonic_or_sigma
        poles[state] = (
            cz_signed[pole_index] * kz * ctx.vtzs[species]
            + kz * ctx.vdsz[species]
            + kx * ctx.vdsx[species]
            - 1j * ctx.nus[species]
            + harmonic * ctx.wcs[species]
        )
        key = (species, harmonic)
        if key not in abc_cache:
            abc_cache[key] = _em_magnetized_abc_terms(
                ctx,
                species=species,
                harmonic=harmonic,
                a_values=a_values[:, species],
                quadrature_order=quadrature_order,
            )
        residues[state] = _em_magnetized_residue(
            ctx,
            species=species,
            harmonic=harmonic,
            pole_index=pole_index,
            pole=poles[state],
            kx=kx,
            kz=kz,
            abc=abc_cache[key],
            cz_signed=cz_signed,
        )

    constant = -np.sum(residues, axis=0)
    diagonal_correction = float(np.sum(ctx.wps2))
    constant[np.diag_indices(3)] -= diagonal_correction
    return LegacyCoefficients(
        poles=_readonly(poles),
        em_residues=_readonly(residues),
        em_constant=_readonly(constant),
    )

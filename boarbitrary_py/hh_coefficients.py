from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coeffs import fun_in, precompute_abc
from .config import MatrixDiagnostics, ScanPoint, SolverContext


@dataclass(frozen=True)
class HHMatrixCoefficients:
    csnj: np.ndarray
    bsnj: np.ndarray
    bs: np.ndarray
    snj_species: np.ndarray
    snj_harmonic: np.ndarray
    snj_pole: np.ndarray
    kx: float
    kx_raw: float
    kz: float
    k: float
    c2: float
    s_count: int
    snj: int
    snj1: int
    snj3: int
    nn: int
    iem: int
    quadrature_order: int

    def validate(self) -> None:
        if self.csnj.shape != (self.snj,):
            raise ValueError(f"csnj shape mismatch: {self.csnj.shape} != {(self.snj,)}")
        if self.bsnj.shape != (self.snj, 3, 3):
            raise ValueError(f"bsnj shape mismatch: {self.bsnj.shape} != {(self.snj, 3, 3)}")
        if self.bs.shape != (self.s_count, 3, 3):
            raise ValueError(f"bs shape mismatch: {self.bs.shape} != {(self.s_count, 3, 3)}")
        if self.snj_species.shape != (self.snj,):
            raise ValueError(
                f"snj_species shape mismatch: {self.snj_species.shape} != {(self.snj,)}"
            )
        if self.snj_harmonic.shape != (self.snj,):
            raise ValueError(
                f"snj_harmonic shape mismatch: {self.snj_harmonic.shape} != {(self.snj,)}"
            )
        if self.snj_pole.shape != (self.snj,):
            raise ValueError(f"snj_pole shape mismatch: {self.snj_pole.shape} != {(self.snj,)}")
        if self.snj1 != self.snj + self.s_count:
            raise ValueError("snj1 must equal snj + s_count")
        if self.snj3 != 3 * self.snj1:
            raise ValueError("snj3 must equal 3 * snj1")
        if self.nn != self.snj3 + 6:
            raise ValueError("nn must equal snj3 + 6")

    def to_diagnostics(self) -> MatrixDiagnostics:
        return MatrixDiagnostics(
            csnj=self.csnj,
            b11s=self.bs[:, 0, 0],
            b22s=self.bs[:, 1, 1],
            b33s=self.bs[:, 2, 2],
        )


def compute_hh_coefficients(
    ctx: SolverContext,
    point: ScanPoint,
    *,
    quadrature_order: int = 120,
    sum_mode: str = "vectorized",
) -> HHMatrixCoefficients:
    if ctx.setup.iem not in (1, 2):
        raise NotImplementedError("Only HH electromagnetic/Darwin matrix assembly is in scope.")

    kx = point.kx if point.kx != 0 else 1e-30
    kz = point.kz
    k = float(np.sqrt(kz**2 + kx**2))
    a_s = kx * ctx.rhocs * np.sqrt(2.0)

    czjj = -ctx.czj if kz < 0 else ctx.czj
    czjj_l = np.zeros((len(czjj), int(np.max(ctx.lsmax)) + 5), dtype=np.complex128)
    for ell in range(int(np.max(ctx.lsmax)) + 4):
        czjj_l[:, ell + 1] = czjj**ell

    b11s = np.zeros(ctx.s_count, dtype=np.complex128)
    b12s = np.zeros_like(b11s)
    b13s = np.zeros_like(b11s)
    b21s = np.zeros_like(b11s)
    b22s = np.zeros_like(b11s)
    b23s = np.zeros_like(b11s)
    b31s = np.zeros_like(b11s)
    b32s = np.zeros_like(b11s)
    b33s = np.zeros_like(b11s)

    csnj = np.zeros(ctx.snj, dtype=np.complex128)
    b11snj = np.zeros_like(csnj)
    b12snj = np.zeros_like(csnj)
    b13snj = np.zeros_like(csnj)
    b21snj = np.zeros_like(csnj)
    b22snj = np.zeros_like(csnj)
    b23snj = np.zeros_like(csnj)
    b31snj = np.zeros_like(csnj)
    b32snj = np.zeros_like(csnj)
    b33snj = np.zeros_like(csnj)

    snj_species = np.zeros(ctx.snj, dtype=np.int64)
    snj_harmonic = np.zeros(ctx.snj, dtype=np.int64)
    snj_pole = np.zeros(ctx.snj, dtype=np.int64)

    snj_index = 0
    for s in range(ctx.s_count):
        for n in range(-int(ctx.nss[s]), int(ctx.nss[s]) + 1):
            ans, bns, cns = precompute_abc(
                n,
                float(a_s[s]),
                float(ctx.ds[s]),
                int(ctx.msmax[s]),
                float(ctx.cap_as[s]),
                order=quadrature_order,
            )
            if sum_mode == "vectorized":
                sums = _matrix_sums_all_j(ctx, s, ans, bns, cns, czjj_l)
            elif sum_mode in {"matlab-loop", "matlab_loop"}:
                sums = _matrix_sums_all_j_matlab_loop(ctx, s, ans, bns, cns, czjj_l)
            else:
                raise ValueError(f"Unknown HH coefficient sum_mode: {sum_mode}")
            for j in range(ctx.setup.j_poles):
                cnj = czjj[j] * kz * ctx.vtzs[s] + kz * ctx.vdsz[s] + n * ctx.wcs[s]
                csnj[snj_index] = cnj
                snj_species[snj_index] = s
                snj_harmonic[snj_index] = n
                snj_pole[snj_index] = j

                if j == 0:
                    p11_b = (n * ctx.wcs[s] / (kx * ctx.vtps[s])) ** 2 * sums["sum11tmp3"]
                    p22_b = sums["sum22tmp3"]
                    p33_b = sums["sum33tmp3"]
                    b11s[s] -= ctx.wps2[s] * p11_b
                    b22s[s] -= ctx.wps2[s] * p22_b
                    b33s[s] -= ctx.wps2[s] * p33_b

                tmp = ctx.wps2[s] * ctx.bzj[j] / cnj
                p11 = (n * ctx.wcs[s] / (kx * ctx.vtps[s])) ** 2 * (
                    n * ctx.wcs[s] * sums["sum11tmp1"][j]
                    + (kz * ctx.vtzs[s])
                    * (ctx.vtps[s] / ctx.vtzs[s]) ** 2
                    * sums["sum11tmp2"][j]
                )
                p12 = 1j * (n * ctx.wcs[s] / (kx * ctx.vtps[s])) * (
                    n * ctx.wcs[s] * sums["sum12tmp1"][j]
                    + (kz * ctx.vtzs[s])
                    * (ctx.vtps[s] / ctx.vtzs[s]) ** 2
                    * sums["sum12tmp2"][j]
                )
                p21 = -p12
                p22 = n * ctx.wcs[s] * sums["sum22tmp1"][j] + (
                    kz * ctx.vtzs[s]
                ) * (ctx.vtps[s] / ctx.vtzs[s]) ** 2 * sums["sum22tmp2"][j]
                p13 = (n * ctx.wcs[s] / (kx * ctx.vtps[s])) * (
                    (ctx.vtzs[s] / ctx.vtps[s]) * n * ctx.wcs[s] * sums["sum13tmp1"][j]
                    + (kz * ctx.vtps[s]) * sums["sum13tmp2"][j]
                )
                p31 = p13
                p23 = -1j * (
                    (ctx.vtzs[s] / ctx.vtps[s]) * n * ctx.wcs[s] * sums["sum23tmp1"][j]
                    + (kz * ctx.vtps[s]) * sums["sum23tmp2"][j]
                )
                p32 = 1j * (
                    (ctx.vtzs[s] / ctx.vtps[s]) * n * ctx.wcs[s] * sums["sum32tmp1"][j]
                    + (kz * ctx.vtps[s]) * sums["sum32tmp2"][j]
                )
                p33 = (ctx.vtzs[s] / ctx.vtps[s]) * (
                    (ctx.vtzs[s] / ctx.vtps[s]) * n * ctx.wcs[s] * sums["sum33tmp1"][j]
                    + (kz * ctx.vtps[s]) * sums["sum33tmp2"][j]
                )

                b11snj[snj_index] += tmp * p11
                b11s[s] -= tmp * p11
                b12snj[snj_index] += tmp * p12
                b12s[s] -= tmp * p12
                b21snj[snj_index] += tmp * p21
                b21s[s] -= tmp * p21
                b22snj[snj_index] += tmp * p22
                b22s[s] -= tmp * p22
                b13snj[snj_index] += tmp * p13
                b13s[s] -= tmp * p13
                b31snj[snj_index] += tmp * p31
                b31s[s] -= tmp * p31
                b23snj[snj_index] += tmp * p23
                b23s[s] -= tmp * p23
                b32snj[snj_index] += tmp * p32
                b32s[s] -= tmp * p32
                b33snj[snj_index] += tmp * p33
                b33s[s] -= tmp * p33
                snj_index += 1

    if snj_index != ctx.snj:
        raise RuntimeError(f"Internal SNJ count mismatch: got {snj_index}, expected {ctx.snj}")

    bsnj = np.empty((ctx.snj, 3, 3), dtype=np.complex128)
    bsnj[:, 0, 0] = b11snj
    bsnj[:, 0, 1] = b12snj
    bsnj[:, 0, 2] = b13snj
    bsnj[:, 1, 0] = b21snj
    bsnj[:, 1, 1] = b22snj
    bsnj[:, 1, 2] = b23snj
    bsnj[:, 2, 0] = b31snj
    bsnj[:, 2, 1] = b32snj
    bsnj[:, 2, 2] = b33snj

    bs = np.empty((ctx.s_count, 3, 3), dtype=np.complex128)
    bs[:, 0, 0] = b11s
    bs[:, 0, 1] = b12s
    bs[:, 0, 2] = b13s
    bs[:, 1, 0] = b21s
    bs[:, 1, 1] = b22s
    bs[:, 1, 2] = b23s
    bs[:, 2, 0] = b31s
    bs[:, 2, 1] = b32s
    bs[:, 2, 2] = b33s

    coeff = HHMatrixCoefficients(
        csnj=csnj,
        bsnj=bsnj,
        bs=bs,
        snj_species=snj_species,
        snj_harmonic=snj_harmonic,
        snj_pole=snj_pole,
        kx=float(kx),
        kx_raw=float(point.kx),
        kz=float(kz),
        k=k,
        c2=float(ctx.c2),
        s_count=ctx.s_count,
        snj=ctx.snj,
        snj1=ctx.snj1,
        snj3=ctx.snj3,
        nn=ctx.nn,
        iem=ctx.setup.iem,
        quadrature_order=quadrature_order,
    )
    coeff.validate()
    return coeff


def _matrix_sums_all_j(ctx: SolverContext, s: int, ans, bns, cns, czjj_l):
    vdz_over_vtz = ctx.vdsz[s] / ctx.vtzs[s]
    lmax = int(ctx.lsmax[s])
    msmax = int(ctx.msmax[s])
    ell = np.arange(lmax + 1, dtype=np.float64)
    m = np.arange(msmax + 1, dtype=np.float64)
    aslm = ctx.aslm[s][: lmax + 1, : msmax + 1]

    radial_a = 2 * ans[2 : msmax + 3, 0] - m * ans[: msmax + 1, 0]
    radial_b = 2 * bns[2 : msmax + 3, 0] - m * bns[: msmax + 1, 0]
    radial_c = 2 * cns[2 : msmax + 3, 0] - m * cns[: msmax + 1, 0]
    a_m = ans[1 : msmax + 2, 1]
    b_m = bns[1 : msmax + 2, 1]
    c_m = cns[1 : msmax + 2, 1]

    a_rad_l = aslm @ radial_a
    b_rad_l = aslm @ radial_b
    c_rad_l = aslm @ radial_c
    a_m_l = aslm @ a_m
    b_m_l = aslm @ b_m
    c_m_l = aslm @ c_m

    c_l = czjj_l[:, 1 : lmax + 2]
    c_lm1 = czjj_l[:, : lmax + 1]
    c_lp1 = czjj_l[:, 2 : lmax + 3]
    c_lp2 = czjj_l[:, 3 : lmax + 4]
    c_lp3 = czjj_l[:, 4 : lmax + 5]

    z1 = 2 * c_lp1 - ell * c_lm1
    z2 = 2 * c_lp2 - ell * c_l
    z3 = 2 * c_lp3 - ell * c_lp1

    in_l = np.array([fun_in(int(value)) for value in ell], dtype=np.float64)
    in_lm1 = np.array([fun_in(int(value) - 1) for value in ell], dtype=np.float64)
    in_lp1 = np.array([fun_in(int(value) + 1) for value in ell], dtype=np.float64)
    in_lp2 = np.array([fun_in(int(value) + 2) for value in ell], dtype=np.float64)

    return {
        "sum11tmp1": c_l @ a_rad_l,
        "sum11tmp2": z1 @ a_m_l,
        "sum11tmp3": np.dot(in_l, a_rad_l),
        "sum12tmp1": c_l @ b_rad_l,
        "sum12tmp2": z1 @ b_m_l,
        "sum22tmp1": c_l @ c_rad_l,
        "sum22tmp2": z1 @ c_m_l,
        "sum22tmp3": np.dot(in_l, c_rad_l),
        "sum13tmp1": (vdz_over_vtz * c_l + c_lp1) @ a_rad_l,
        "sum13tmp2": (z2 + vdz_over_vtz * z1) @ a_m_l,
        "sum23tmp1": (vdz_over_vtz * c_l + c_lp1) @ b_rad_l,
        "sum23tmp2": (z2 + vdz_over_vtz * z1) @ b_m_l,
        "sum32tmp1": (vdz_over_vtz * c_l + c_lp1) @ b_rad_l,
        "sum32tmp2": (z2 + vdz_over_vtz * z1) @ b_m_l,
        "sum33tmp1": (vdz_over_vtz**2 * c_l + 2 * vdz_over_vtz * c_lp1 + c_lp2)
        @ a_rad_l,
        "sum33tmp2": (vdz_over_vtz**2 * z1 + 2 * vdz_over_vtz * z2 + z3) @ a_m_l,
        "sum33tmp3": np.dot(
            vdz_over_vtz * (2 * in_lp1 - ell * in_lm1) + (2 * in_lp2 - ell * in_l),
            a_m_l,
        ),
    }


def _matrix_sums_all_j_matlab_loop(ctx: SolverContext, s: int, ans, bns, cns, czjj_l):
    vdz_over_vtz = ctx.vdsz[s] / ctx.vtzs[s]
    lmax = int(ctx.lsmax[s])
    msmax = int(ctx.msmax[s])
    j_poles = int(ctx.setup.j_poles)

    sum11tmp1 = np.zeros(j_poles, dtype=np.complex128)
    sum11tmp2 = np.zeros(j_poles, dtype=np.complex128)
    sum12tmp1 = np.zeros(j_poles, dtype=np.complex128)
    sum12tmp2 = np.zeros(j_poles, dtype=np.complex128)
    sum22tmp1 = np.zeros(j_poles, dtype=np.complex128)
    sum22tmp2 = np.zeros(j_poles, dtype=np.complex128)
    sum13tmp1 = np.zeros(j_poles, dtype=np.complex128)
    sum13tmp2 = np.zeros(j_poles, dtype=np.complex128)
    sum23tmp1 = np.zeros(j_poles, dtype=np.complex128)
    sum23tmp2 = np.zeros(j_poles, dtype=np.complex128)
    sum32tmp1 = np.zeros(j_poles, dtype=np.complex128)
    sum32tmp2 = np.zeros(j_poles, dtype=np.complex128)
    sum33tmp1 = np.zeros(j_poles, dtype=np.complex128)
    sum33tmp2 = np.zeros(j_poles, dtype=np.complex128)
    sum11tmp3 = 0.0 + 0.0j
    sum22tmp3 = 0.0 + 0.0j
    sum33tmp3 = 0.0 + 0.0j

    aslm = ctx.aslm[s]
    for j in range(j_poles):
        for ell in range(lmax + 1):
            for m in range(msmax + 1):
                aslm_value = aslm[ell, m]
                a_rad = 2 * ans[m + 2, 0] - m * ans[m, 0]
                b_rad = 2 * bns[m + 2, 0] - m * bns[m, 0]
                c_rad = 2 * cns[m + 2, 0] - m * cns[m, 0]
                a_m = ans[m + 1, 1]
                b_m = bns[m + 1, 1]
                c_m = cns[m + 1, 1]

                c_l = czjj_l[j, ell + 1]
                c_lm1 = czjj_l[j, ell]
                c_lp1 = czjj_l[j, ell + 2]
                c_lp2 = czjj_l[j, ell + 3]
                c_lp3 = czjj_l[j, ell + 4]

                z1 = 2 * c_lp1 - ell * c_lm1
                z2 = 2 * c_lp2 - ell * c_l
                z3 = 2 * c_lp3 - ell * c_lp1

                sum11tmp1[j] += aslm_value * c_l * a_rad
                sum11tmp2[j] += aslm_value * z1 * a_m

                sum12tmp1[j] += aslm_value * c_l * b_rad
                sum12tmp2[j] += aslm_value * z1 * b_m

                sum22tmp1[j] += aslm_value * c_l * c_rad
                sum22tmp2[j] += aslm_value * z1 * c_m

                sum13tmp1[j] += aslm_value * (vdz_over_vtz * c_l + c_lp1) * a_rad
                sum13tmp2[j] += aslm_value * (z2 + vdz_over_vtz * z1) * a_m

                sum23tmp1[j] += aslm_value * (vdz_over_vtz * c_l + c_lp1) * b_rad
                sum23tmp2[j] += aslm_value * (z2 + vdz_over_vtz * z1) * b_m

                sum32tmp1[j] += aslm_value * (vdz_over_vtz * c_l + c_lp1) * b_rad
                sum32tmp2[j] += aslm_value * (z2 + vdz_over_vtz * z1) * b_m

                sum33tmp1[j] += aslm_value * (
                    vdz_over_vtz**2 * c_l + 2 * vdz_over_vtz * c_lp1 + c_lp2
                ) * a_rad
                sum33tmp2[j] += aslm_value * (
                    vdz_over_vtz**2 * z1 + 2 * vdz_over_vtz * z2 + z3
                ) * a_m

                if j == 0:
                    sum11tmp3 += aslm_value * fun_in(ell) * a_rad
                    sum22tmp3 += aslm_value * fun_in(ell) * c_rad
                    sum33tmp3 += aslm_value * (
                        vdz_over_vtz * (2 * fun_in(ell + 1) - ell * fun_in(ell - 1))
                        + (2 * fun_in(ell + 2) - ell * fun_in(ell))
                    ) * a_m

    return {
        "sum11tmp1": sum11tmp1,
        "sum11tmp2": sum11tmp2,
        "sum11tmp3": sum11tmp3,
        "sum12tmp1": sum12tmp1,
        "sum12tmp2": sum12tmp2,
        "sum22tmp1": sum22tmp1,
        "sum22tmp2": sum22tmp2,
        "sum22tmp3": sum22tmp3,
        "sum13tmp1": sum13tmp1,
        "sum13tmp2": sum13tmp2,
        "sum23tmp1": sum23tmp1,
        "sum23tmp2": sum23tmp2,
        "sum32tmp1": sum32tmp1,
        "sum32tmp2": sum32tmp2,
        "sum33tmp1": sum33tmp1,
        "sum33tmp2": sum33tmp2,
        "sum33tmp3": sum33tmp3,
    }

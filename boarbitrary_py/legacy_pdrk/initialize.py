from __future__ import annotations

from math import floor

import numpy as np
from scipy.special import erfc

from ..jpole import j_pole_coefficients
from .config import (
    LegacyCaseInputs,
    LegacyContext,
    LegacyIndexMap,
    LegacyPhysicalConstants,
)


def constants_for_system(name: str) -> LegacyPhysicalConstants:
    if name == "si_2018":
        c2 = (2.99792458e8) ** 2
        epsilon0 = 8.854187817e-12
        return LegacyPhysicalConstants(
            c2=c2,
            epsilon0=epsilon0,
            mu0=1.0 / (c2 * epsilon0),
            kb=1.38064852e-23,
            qe=1.60217662e-19,
            mp=1.6726219e-27,
            me=9.1094e-31,
            munit=1.6726219e-27,
        )
    if name == "muschietti_normalized":
        c2 = 300.0**2
        return LegacyPhysicalConstants(
            c2=c2,
            epsilon0=1.0 / c2,
            mu0=1.0,
            kb=1.0,
            qe=1.0,
            mp=1.0,
            me=1.0 / 900.0,
            munit=1.0,
        )
    raise ValueError(f"Unsupported legacy constant system: {name}")


def _matlab_round_positive(value: float) -> int:
    if value < 0:
        raise ValueError("legacy scan count expects a nonnegative interval")
    return int(floor(value + 0.5))


def _scan_arrays(inputs: LegacyCaseInputs) -> tuple[np.ndarray, np.ndarray]:
    scan = inputs.config.scan
    npa = _matlab_round_positive((scan.pa2 - scan.pa1) / scan.dpa) + 1
    pa = scan.pa1 + np.arange(npa, dtype=float) * scan.dpa
    if scan.ipa == scan.ipb:
        pb = np.asarray([pa[0]], dtype=float)
    else:
        npb = _matlab_round_positive((scan.pb2 - scan.pb1) / scan.dpb) + 1
        pb = scan.pb1 + np.arange(npb, dtype=float) * scan.dpb
    return pa, pb


def _normalization(
    recipe: str,
    *,
    wcs: np.ndarray,
    wps: np.ndarray,
    wps2: np.ndarray,
    lambda_ds: np.ndarray,
    c: float,
) -> tuple[float, float]:
    wn = float(abs(wcs[0]))
    if recipe == "first_species_cyclotron_and_plasma_skin":
        return wn, float(wps[0] / c)
    if recipe == "first_species_cyclotron_skin":
        return wn, float(abs(wcs[0]) / c)
    if recipe == "total_plasma_and_combined_debye":
        return float(np.sqrt(np.sum(wps2))), float(np.sqrt(np.sum(1.0 / lambda_ds**2)))
    if recipe == "first_species_cyclotron_and_debye":
        return wn, float(1.0 / lambda_ds[0])
    if recipe == "first_species_cyclotron_and_two_ion_plasma_skin":
        return wn, float(np.sqrt(wps2[0] + wps2[1]) / c)
    if recipe == "first_species_cyclotron_and_combined_debye":
        return wn, float(np.sqrt(np.sum(1.0 / lambda_ds**2)))
    raise ValueError(f"Unsupported legacy normalization recipe: {recipe}")


def _build_index_map(
    *,
    ns_per_species: np.ndarray,
    nss: np.ndarray,
    imus: np.ndarray,
    j_poles: int,
) -> LegacyIndexMap:
    species: list[int] = []
    harmonic_or_sigma: list[int] = []
    pole: list[int] = []
    unmagnetized: list[bool] = []
    for s in range(len(ns_per_species)):
        values = range(1, int(ns_per_species[s]) + 1) if not imus[s] else range(-int(nss[s]), int(nss[s]) + 1)
        for n in values:
            for j in range(j_poles):
                species.append(s)
                harmonic_or_sigma.append(n)
                pole.append(j)
                unmagnetized.append(not bool(imus[s]))
    return LegacyIndexMap(
        species=np.asarray(species, dtype=np.int32),
        harmonic_or_sigma=np.asarray(harmonic_or_sigma, dtype=np.int32),
        pole=np.asarray(pole, dtype=np.int32),
        unmagnetized=np.asarray(unmagnetized, dtype=bool),
    )


def _legacy_j_pole_coefficients(j_poles: int, variant: str) -> tuple[np.ndarray, np.ndarray]:
    if variant == "j8_i10_opt3":
        return j_pole_coefficients(j_poles)
    if variant == "j8_i8_opt2":
        if j_poles != 8:
            raise ValueError(f"J-pole variant {variant} requires J=8, got J={j_poles}")
        bzj_half = np.asarray(
            [
                2.262756990456044 + 0.574863654552974j,
                0.007755112891899 + 0.435320280255620j,
                -0.017631167787210 - 0.001036701965463j,
                -2.752880935560734 - 4.080667522419399j,
            ],
            dtype=np.complex128,
        )
        czj_half = np.asarray(
            [
                -0.845779591068908 - 1.626106289905113j,
                1.485551642694134 - 1.512332910786984j,
                2.286671582218345 - 1.340185584073022j,
                0.275217303800922 - 1.682797352333194j,
            ],
            dtype=np.complex128,
        )
        return (
            np.concatenate((bzj_half, np.conjugate(bzj_half))),
            np.concatenate((czj_half, -np.conjugate(czj_half))),
        )
    raise ValueError(f"Unsupported legacy J-pole variant: {variant}")


def initialize_legacy_case(inputs: LegacyCaseInputs) -> LegacyContext:
    case = inputs.config
    model = case.model
    if model.sp != 0:
        raise NotImplementedError("The legacy compatibility milestone currently requires sp=0.")
    if model.iem not in {0, 1, 2}:
        raise ValueError(f"Unsupported legacy iem={model.iem}")
    if model.set_ns_semantics != "legacy_row_index":
        raise ValueError(f"Unsupported setNs semantics: {model.set_ns_semantics}")

    constants = constants_for_system(case.constant_system)
    species_inputs = inputs.species
    s_count = len(species_inputs)
    qs0 = np.asarray([item.charge_qe for item in species_inputs], dtype=float)
    ms0 = np.asarray([item.mass_munit for item in species_inputs], dtype=float)
    ns0 = np.asarray([item.density for item in species_inputs], dtype=float)
    tzs0 = np.asarray([item.tz_ev for item in species_inputs], dtype=float)
    tps0 = np.asarray([item.tp_ev for item in species_inputs], dtype=float)
    alphas = np.asarray([item.alpha for item in species_inputs], dtype=float)
    deltas = np.asarray([item.delta for item in species_inputs], dtype=float)
    vdsz0 = np.asarray([item.drift_z_c for item in species_inputs], dtype=float)
    vdsx0 = np.asarray([item.drift_x_c for item in species_inputs], dtype=float)
    vdsy0 = np.asarray([item.drift_y_c for item in species_inputs], dtype=float)
    vdsr0 = np.asarray([item.drift_r_c for item in species_inputs], dtype=float)
    nus0 = np.asarray([item.collision_frequency for item in species_inputs], dtype=float)
    imus = np.asarray([item.magnetized for item in species_inputs], dtype=bool)
    if case.b0 == 0.0:
        imus = np.zeros_like(imus)

    nrsab = np.full(s_count, 2, dtype=np.int32)
    rsab = np.zeros((2, s_count), dtype=float)
    for s in range(s_count):
        if alphas[s] == 1.0 or deltas[s] == 1.0:
            rsab[:, s] = (1.0, 0.0)
            nrsab[s] = 1
        else:
            rsab[0, s] = (1.0 - alphas[s] * deltas[s]) / (1.0 - alphas[s])
            rsab[1, s] = alphas[s] * (deltas[s] - 1.0) / (1.0 - alphas[s])

    qs = qs0 * constants.qe
    ms = ms0 * constants.munit
    tzs = tzs0 * constants.qe / constants.kb
    tps = tps0 * constants.qe / constants.kb
    vdsz = vdsz0 * constants.c
    vdsx = vdsx0 * constants.c
    vdsy = vdsy0 * constants.c
    vdsr = vdsr0 * constants.c
    vdsr = vdsr.copy()
    vdsr[~imus] = 0.0
    nus = nus0.copy()

    ns_per_species = np.full(s_count, 2 * model.n_harmonics + 1, dtype=np.int32)
    # Reproduce BO v190305 exactly: setNs is applied by row number, not its declared species ID.
    for row_index, (declared_species, n_value) in enumerate(model.set_ns):
        if declared_species <= s_count and row_index < s_count:
            ns_per_species[row_index] = 2 * n_value + 1
    ns_per_species[~imus] = nrsab[~imus]
    nss = np.floor((ns_per_species - 1) / 2).astype(np.int32)

    vtzs = np.sqrt(2.0 * constants.kb * tzs / ms)
    vtps = np.sqrt(2.0 * constants.kb * tps / ms)
    lambda_ds = np.sqrt(constants.epsilon0 * constants.kb * tzs / (ns0 * qs**2))
    kds = 1.0 / lambda_ds
    wps = np.sqrt(ns0 * qs**2 / (ms * constants.epsilon0))
    wps2 = wps**2
    wcs = case.b0 * qs / ms
    rhocs = np.sqrt(constants.kb * tps / ms) / wcs
    lmd_t = tzs / tps

    tpsab = np.vstack([tps, alphas * tps])
    vtpsab = np.sqrt(2.0 * constants.kb * tpsab / ms[None, :])
    lmd_tab = tzs[None, :] / tpsab
    rhocsab = np.sqrt(constants.kb * tpsab / ms[None, :]) / wcs[None, :]
    bsab = vdsr[None, :] / vtpsab
    asab_norm = np.exp(-(bsab**2)) + np.sqrt(np.pi) * bsab * erfc(-bsab)

    beta_sz = 2.0 * constants.mu0 * constants.kb * ns0 * tzs / case.b0**2
    beta_sp = 2.0 * constants.mu0 * constants.kb * ns0 * tps / case.b0**2
    va = float(case.b0 / np.sqrt(constants.mu0 * np.sum(ms * ns0)))
    wn, kn = _normalization(
        case.normalization_recipe,
        wcs=wcs,
        wps=wps,
        wps2=wps2,
        lambda_ds=lambda_ds,
        c=constants.c,
    )

    bzj, czj = _legacy_j_pole_coefficients(model.j_poles, model.jpole_variant)
    snj = int(np.sum(ns_per_species) * model.j_poles)
    snj1 = snj + 1
    snj3 = 3 * snj1
    nn = snj1 if model.iem == 0 else snj3 + 6
    pa, pb = _scan_arrays(inputs)
    index_map = _build_index_map(
        ns_per_species=ns_per_species,
        nss=nss,
        imus=imus,
        j_poles=model.j_poles,
    )
    if len(index_map.species) != snj:
        raise AssertionError("Legacy state index map does not match SNJ")
    if len(pa) * len(pb) != case.canonical.scan_points:
        raise ValueError(
            f"Canonical scan count mismatch for {case.case_id}: "
            f"computed {len(pa) * len(pb)}, expected {case.canonical.scan_points}"
        )
    if nn != case.canonical.matrix_size:
        raise ValueError(
            f"Canonical matrix size mismatch for {case.case_id}: computed {nn}, "
            f"expected {case.canonical.matrix_size}"
        )

    return LegacyContext(
        case=case,
        species_inputs=species_inputs,
        constants=constants,
        s_count=s_count,
        qs0=qs0,
        ms0=ms0,
        ns0=ns0,
        tzs0=tzs0,
        tps0=tps0,
        alphas=alphas,
        deltas=deltas,
        vdsz0=vdsz0,
        vdsx0=vdsx0,
        vdsy0=vdsy0,
        vdsr0=vdsr0,
        nus0=nus0,
        imus=imus,
        nrsab=nrsab,
        rsab=rsab,
        qs=qs,
        ms=ms,
        tzs=tzs,
        tps=tps,
        vdsz=vdsz,
        vdsx=vdsx,
        vdsy=vdsy,
        vdsr=vdsr,
        nus=nus,
        vtzs=vtzs,
        vtps=vtps,
        lambda_ds=lambda_ds,
        kds=kds,
        wps=wps,
        wps2=wps2,
        wcs=wcs,
        rhocs=rhocs,
        lmd_t=lmd_t,
        tpsab=tpsab,
        vtpsab=vtpsab,
        lmd_tab=lmd_tab,
        rhocsab=rhocsab,
        bsab=bsab,
        asab_norm=asab_norm,
        beta_sz=beta_sz,
        beta_sp=beta_sp,
        va=va,
        charge_total_normalized=float(np.sum(qs0 * ns0)),
        current_totals_normalized=np.asarray(
            [np.sum(qs0 * ns0 * vdsz0), np.sum(qs0 * ns0 * vdsx0), np.sum(qs0 * ns0 * vdsy0)]
        ),
        wn=wn,
        kn=kn,
        ns_per_species=ns_per_species,
        nss=nss,
        bzj=bzj,
        czj=czj,
        snj=snj,
        snj1=snj1,
        snj3=snj3,
        nn=nn,
        nw=nn,
        pa=pa,
        pb=pb,
        index_map=index_map,
    )

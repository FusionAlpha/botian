from __future__ import annotations

import math
import re

import numpy as np
from scipy.special import erfc

from .config import CaseInputs, SolverContext
from .io import load_case_inputs
from .jpole import j_pole_coefficients


def initialize_case(case_inputs: CaseInputs | None = None) -> SolverContext:
    case_inputs = case_inputs or load_case_inputs()
    setup = case_inputs.setup
    species = case_inputs.bo_input.species
    s_count = len(species)

    c2 = setup.c2
    epsilon0 = setup.epsilon0
    mu0 = setup.mu0
    kb = setup.kb
    qe = setup.qe
    mp = setup.mp
    me = setup.me
    munit = setup.munit

    qs0 = np.array([s.charge_qe for s in species], dtype=float)
    ms0 = np.array([s.mass_munit for s in species], dtype=float)
    ns0 = np.array([s.density for s in species], dtype=float)
    tzs0 = np.array([s.tz_ev for s in species], dtype=float)
    tps0 = np.array([s.tp_ev for s in species], dtype=float)
    vdsz0 = np.array([s.drift_z_c for s in species], dtype=float)
    vdsr0 = np.array([s.drift_r_c for s in species], dtype=float)
    ifv0 = np.array([s.ifv for s in species], dtype=int)

    qs = qs0 * qe
    ms = ms0 * munit
    aslm: list[np.ndarray] = []
    lsmax = np.zeros(s_count, dtype=int)
    msmax = np.zeros(s_count, dtype=int)
    c = math.sqrt(c2)

    for idx in range(s_count):
        if ifv0[idx] == 1:
            aslm.append(np.ones((1, 1), dtype=np.complex128))
        else:
            fvc = case_inputs.fv_coefficients[idx + 1]
            aslm.append(np.asarray(fvc.alm, dtype=np.complex128))
            vdsz0[idx] = fvc.vdz / c
            vdsr0[idx] = fvc.vdr / c
            tzs0[idx] = ms[idx] * fvc.vtz**2 / (2.0 * qe)
            tps0[idx] = ms[idx] * fvc.vtp**2 / (2.0 * qe)
        lsmax[idx] = aslm[-1].shape[0] - 1
        msmax[idx] = aslm[-1].shape[1] - 1

    tzs = tzs0 * qe / kb
    tps = tps0 * qe / kb
    vdsz = vdsz0 * c
    vdsr = vdsr0 * c

    n = setup.n_harmonics
    ns_per_species = np.full(s_count, 2 * n + 1, dtype=int)
    nss = ((ns_per_species - 1) // 2).astype(int)

    vtzs = np.sqrt(2.0 * kb * tzs / ms)
    vtps = np.sqrt(2.0 * kb * tps / ms)
    lambda_ds = np.sqrt(epsilon0 * kb * tzs / (ns0 * qs**2))
    wps = np.sqrt(ns0 * qs**2 / ms / epsilon0)
    wps2 = wps**2
    wcs = setup.b0 * qs / ms
    rhocs = np.sqrt(kb * tps / ms) / wcs
    lmd_t = tzs / tps
    ds = vdsr / vtps
    cap_as = np.exp(-(ds**2)) + math.sqrt(math.pi) * ds * erfc(-ds)

    beta_sz = 2.0 * mu0 * kb * ns0 * tzs / setup.b0**2
    beta_sp = 2.0 * mu0 * kb * ns0 * tps / setup.b0**2
    va = setup.b0 / math.sqrt(mu0 * float(np.sum(ms * ns0)))
    lambda_d = math.sqrt(1.0 / float(np.sum(1.0 / (lambda_ds**2))))
    wn, kn = _normalization_from_bo_knwn(
        case_inputs.paths.bo_knwn,
        default_wn=abs(wcs[0]),
        default_kn=abs(wcs[0]) / va,
        lambda_d=lambda_d,
        lambda_ds=lambda_ds,
        wcs=wcs,
        wps=wps,
        rhocs=rhocs,
        va=va,
        c2=c2,
    )

    bzj, czj = j_pole_coefficients(setup.j_poles)

    snj = int(np.sum(ns_per_species) * setup.j_poles)
    snj1 = snj + s_count
    snj3 = 3 * snj1
    nn = snj3 + 6
    ids0 = np.zeros(s_count, dtype=int)
    ids1 = np.zeros(s_count, dtype=int)
    ids2 = np.zeros(s_count, dtype=int)
    for idx in range(s_count):
        ids0[idx] = int(np.sum(ns_per_species[:idx]) * setup.j_poles)
        ids1[idx] = ids0[idx] + ns_per_species[idx] * setup.j_poles - 1
        ids2[idx] = snj + idx

    npa = round((setup.pa2 - setup.pa1) / setup.dpa) + 1
    pa = setup.pa1 + np.arange(npa, dtype=float) * setup.dpa
    if setup.ipa == setup.ipb:
        pb = np.array([pa[0]], dtype=float)
    else:
        npb = round((setup.pb2 - setup.pb1) / setup.dpb) + 1
        pb = setup.pb1 + np.arange(npb, dtype=float) * setup.dpb

    nw = nn if setup.sp == 0 else 1

    return SolverContext(
        paths=case_inputs.paths,
        setup=setup,
        s_count=s_count,
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
        lambda_ds=lambda_ds,
        wps=wps,
        wps2=wps2,
        wcs=wcs,
        rhocs=rhocs,
        lmd_t=lmd_t,
        ds=ds,
        cap_as=cap_as,
        beta_sz=beta_sz,
        beta_sp=beta_sp,
        va=va,
        wn=wn,
        kn=kn,
        ns_per_species=ns_per_species,
        nss=nss,
        lsmax=lsmax,
        msmax=msmax,
        aslm=tuple(aslm),
        bzj=bzj,
        czj=czj,
        snj=snj,
        snj1=snj1,
        snj3=snj3,
        nn=nn,
        ids0=ids0,
        ids1=ids1,
        ids2=ids2,
        pa=pa,
        pb=pb,
        nw=nw,
        c2=c2,
        epsilon0=epsilon0,
        mu0=mu0,
        kb=kb,
        qe=qe,
        mp=mp,
        me=me,
        munit=munit,
    )


def _normalization_from_bo_knwn(
    path,
    *,
    default_wn: float,
    default_kn: float,
    lambda_d: float,
    lambda_ds: np.ndarray,
    wcs: np.ndarray,
    wps: np.ndarray,
    rhocs: np.ndarray,
    va: float,
    c2: float,
) -> tuple[float, float]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return float(default_wn), float(default_kn)
    active = "\n".join(line.split("%", 1)[0] for line in text.splitlines())
    wn_expr = _last_assignment_expression(active, "wn")
    kn_expr = _last_assignment_expression(active, "kn")
    variables = {
        "lambdaD": float(lambda_d),
        "lambdaDs": np.asarray(lambda_ds, dtype=float),
        "wcs": np.asarray(wcs, dtype=float),
        "wps": np.asarray(wps, dtype=float),
        "wcs1": float(abs(wcs[0])),
        "wps1": float(wps[0]),
        "cwp": float(math.sqrt(c2) / wps[0]),
        "vA": float(va),
        "rhocs": np.asarray(rhocs, dtype=float),
    }
    wn = _eval_normalization_expression(wn_expr, variables, default_wn)
    variables["wn"] = wn
    kn = _eval_normalization_expression(kn_expr, variables, default_kn)
    return float(wn), float(kn)


def _last_assignment_expression(text: str, name: str) -> str | None:
    matches = re.findall(rf"(?<![A-Za-z0-9_]){name}\s*=\s*([^;]+);", text)
    return matches[-1].strip() if matches else None


def _eval_normalization_expression(expr: str | None, variables: dict[str, object], default: float) -> float:
    if not expr:
        return float(default)
    compact = re.sub(r"\s+", "", expr)
    if compact in {"abs(wcs(1))", "wcs1"}:
        return float(variables["wcs1"])
    if compact == "wps1":
        return float(variables["wps1"])
    if compact == "wn/vA":
        return float(variables["wn"]) / float(variables["vA"])
    if compact == "1/lambdaD":
        return 1.0 / float(variables["lambdaD"])
    if compact == "1/lambdaDs(1)":
        return 1.0 / float(np.asarray(variables["lambdaDs"])[0])
    if compact == "1/max(abs(rhocs))":
        return 1.0 / float(np.max(np.abs(np.asarray(variables["rhocs"]))))
    if compact == "cwp":
        return float(variables["cwp"])
    if compact == "1/cwp":
        return 1.0 / float(variables["cwp"])
    try:
        return float(compact)
    except ValueError as exc:
        raise NotImplementedError(f"Unsupported bo_knwn normalization expression: {expr}") from exc

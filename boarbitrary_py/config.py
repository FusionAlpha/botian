from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CasePaths:
    root: Path
    bohh_root: Path
    input_dir: Path
    modules_dir: Path
    genfv_dir: Path
    output_dir: Path
    bo_in: Path
    bo_setup: Path
    bo_knwn: Path
    reference_mat: Path | None


@dataclass(frozen=True)
class SetupConfig:
    savepath: str
    iem: int
    n_harmonics: int
    j_poles: int
    sp: int
    wg0: complex
    b0: float
    par: np.ndarray
    ipa: int
    ipb: int
    iloga: int
    ilogb: int
    pa1: float
    pa2: float
    dpa: float
    pb1: float
    pb2: float
    dpb: float
    iout: int
    c2: float
    epsilon0: float
    mu0: float
    kb: float
    qe: float
    mp: float
    me: float
    munit: float


@dataclass(frozen=True)
class SpeciesInput:
    charge_qe: float
    mass_munit: float
    density: float
    tz_ev: float
    tp_ev: float
    drift_z_c: float
    drift_r_c: float
    ifv: int


@dataclass(frozen=True)
class BOInput:
    species: tuple[SpeciesInput, ...]


@dataclass(frozen=True)
class FVCoefficients:
    alm: np.ndarray
    vdz: float
    vdr: float
    vtz: float
    vtp: float


@dataclass(frozen=True)
class CaseInputs:
    paths: CasePaths
    bo_input: BOInput
    setup: SetupConfig
    fv_coefficients: dict[int, FVCoefficients]


@dataclass(frozen=True)
class SolverContext:
    paths: CasePaths
    setup: SetupConfig
    s_count: int
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
    lambda_ds: np.ndarray
    wps: np.ndarray
    wps2: np.ndarray
    wcs: np.ndarray
    rhocs: np.ndarray
    lmd_t: np.ndarray
    ds: np.ndarray
    cap_as: np.ndarray
    beta_sz: np.ndarray
    beta_sp: np.ndarray
    va: float
    wn: float
    kn: float
    ns_per_species: np.ndarray
    nss: np.ndarray
    lsmax: np.ndarray
    msmax: np.ndarray
    aslm: tuple[np.ndarray, ...]
    bzj: np.ndarray
    czj: np.ndarray
    snj: int
    snj1: int
    snj3: int
    nn: int
    ids0: np.ndarray
    ids1: np.ndarray
    ids2: np.ndarray
    pa: np.ndarray
    pb: np.ndarray
    nw: int
    c2: float
    epsilon0: float
    mu0: float
    kb: float
    qe: float
    mp: float
    me: float
    munit: float


@dataclass(frozen=True)
class ScanPoint:
    jpa: int
    jpb: int
    pa_value: float
    pb_value: float
    par: np.ndarray
    k_norm: float
    theta_deg: float
    kz_norm: float
    kx_norm: float
    k: float
    kz: float
    kx: float


@dataclass(frozen=True)
class MatrixDiagnostics:
    csnj: np.ndarray
    b11s: np.ndarray
    b22s: np.ndarray
    b33s: np.ndarray


@dataclass(frozen=True)
class ScanResult:
    ww: np.ndarray
    pa: np.ndarray
    pb: np.ndarray
    kxx: np.ndarray
    kzz: np.ndarray
    kk: np.ndarray
    tt: np.ndarray
    elapsed_seconds: float
    pa_used: np.ndarray | None = None
    pb_used: np.ndarray | None = None
    pa_full: np.ndarray | None = None
    pb_full: np.ndarray | None = None
    scan_indices: np.ndarray | None = None

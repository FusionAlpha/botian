from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class LegacyPhysicalConstants:
    c2: float
    epsilon0: float
    mu0: float
    kb: float
    qe: float
    mp: float
    me: float
    munit: float

    @property
    def c(self) -> float:
        return float(np.sqrt(self.c2))


@dataclass(frozen=True)
class LegacySpeciesInput:
    charge_qe: float
    mass_munit: float
    density: float
    tz_ev: float
    tp_ev: float
    alpha: float
    delta: float
    drift_z_c: float
    drift_x_c: float
    drift_y_c: float
    drift_r_c: float
    collision_frequency: float
    magnetized: bool


@dataclass(frozen=True)
class LegacyModelConfig:
    iem: int
    n_harmonics: int
    j_poles: int
    jpole_variant: str
    sp: int
    set_ns: tuple[tuple[int, int], ...]
    set_ns_semantics: str


@dataclass(frozen=True)
class LegacyScanConfig:
    initial_k: float
    initial_theta_deg: float
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


@dataclass(frozen=True)
class LegacyCanonicalConfig:
    scan_points: int
    coordinate_dimensions: int
    matrix_size: int
    roots_per_scan: int
    overrides: dict[str, Any]


@dataclass(frozen=True)
class LegacyCaseConfig:
    schema_version: int
    case_id: str
    source_family: str
    source_example: str
    root: Path
    raw_input_dir: Path
    model: LegacyModelConfig
    b0: float
    constant_system: str
    scan: LegacyScanConfig
    normalization_recipe: str
    canonical: LegacyCanonicalConfig
    development_oracle: str


@dataclass(frozen=True)
class LegacyCaseInputs:
    config: LegacyCaseConfig
    species: tuple[LegacySpeciesInput, ...]


@dataclass(frozen=True)
class LegacyIndexMap:
    species: np.ndarray
    harmonic_or_sigma: np.ndarray
    pole: np.ndarray
    unmagnetized: np.ndarray


@dataclass(frozen=True)
class LegacyContext:
    case: LegacyCaseConfig
    species_inputs: tuple[LegacySpeciesInput, ...]
    constants: LegacyPhysicalConstants
    s_count: int
    qs0: np.ndarray
    ms0: np.ndarray
    ns0: np.ndarray
    tzs0: np.ndarray
    tps0: np.ndarray
    alphas: np.ndarray
    deltas: np.ndarray
    vdsz0: np.ndarray
    vdsx0: np.ndarray
    vdsy0: np.ndarray
    vdsr0: np.ndarray
    nus0: np.ndarray
    imus: np.ndarray
    nrsab: np.ndarray
    rsab: np.ndarray
    qs: np.ndarray
    ms: np.ndarray
    tzs: np.ndarray
    tps: np.ndarray
    vdsz: np.ndarray
    vdsx: np.ndarray
    vdsy: np.ndarray
    vdsr: np.ndarray
    nus: np.ndarray
    vtzs: np.ndarray
    vtps: np.ndarray
    lambda_ds: np.ndarray
    kds: np.ndarray
    wps: np.ndarray
    wps2: np.ndarray
    wcs: np.ndarray
    rhocs: np.ndarray
    lmd_t: np.ndarray
    tpsab: np.ndarray
    vtpsab: np.ndarray
    lmd_tab: np.ndarray
    rhocsab: np.ndarray
    bsab: np.ndarray
    asab_norm: np.ndarray
    beta_sz: np.ndarray
    beta_sp: np.ndarray
    va: float
    charge_total_normalized: float
    current_totals_normalized: np.ndarray
    wn: float
    kn: float
    ns_per_species: np.ndarray
    nss: np.ndarray
    bzj: np.ndarray
    czj: np.ndarray
    snj: int
    snj1: int
    snj3: int
    nn: int
    nw: int
    pa: np.ndarray
    pb: np.ndarray
    index_map: LegacyIndexMap


@dataclass(frozen=True)
class LegacyScanPoint:
    flat_index: int
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
class LegacyCoefficients:
    poles: np.ndarray
    es_residues: np.ndarray | None = None
    em_residues: np.ndarray | None = None
    em_constant: np.ndarray | None = None


@dataclass(frozen=True)
class LegacyMatrix:
    matrix: csr_matrix
    mass_matrix: csr_matrix | None
    coefficients: LegacyCoefficients


@dataclass(frozen=True)
class LegacyMinimalGroup:
    pole: complex
    coefficient_indices: np.ndarray
    block_size: int
    rank: int
    known_pole_multiplicity: int
    singular_values: np.ndarray
    rank_threshold: float
    rank_gap: float
    rank_ambiguous: bool
    residue_frobenius_norm: float


@dataclass(frozen=True)
class LegacyMcMillanCertificate:
    grouping_mode: str
    original_size: int
    original_auxiliary_size: int
    field_size: int
    pole_group_count: int
    minimal_auxiliary_size: int
    minimal_size: int
    known_pole_count: int
    reconstructed_root_count: int
    count_closure: bool
    ambiguous_rank_count: int
    rank_tolerance: float
    mcmillan_minimal: bool


@dataclass(frozen=True)
class LegacyMinimalRealization:
    matrix: np.ndarray
    dynamic_poles: np.ndarray
    input_factor: np.ndarray
    output_factor: np.ndarray
    field_block: np.ndarray
    known_poles: np.ndarray
    groups: tuple[LegacyMinimalGroup, ...]
    certificate: LegacyMcMillanCertificate


@dataclass(frozen=True)
class LegacyMinimalSpectrum:
    roots: np.ndarray
    dynamic_roots: np.ndarray
    structural_poles: np.ndarray
    realization: LegacyMinimalRealization
    build_seconds: float
    eig_seconds: float


@dataclass(frozen=True)
class LegacyScanResult:
    roots: np.ndarray
    scan_indices: np.ndarray
    k: np.ndarray
    kx: np.ndarray
    kz: np.ndarray
    theta_deg: np.ndarray
    elapsed_seconds: float
    backend: str = "full"
    reduced_matrix_sizes: np.ndarray | None = None
    structural_root_counts: np.ndarray | None = None

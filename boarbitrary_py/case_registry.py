from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


SourceFamily = Literal["2018_bo_pdrk", "2025_bo_arbitrary_hh"]


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    iem: int
    n_harmonics: int
    j_poles: int
    sp: int = 0
    jpole_variant: str | None = None


@dataclass(frozen=True)
class ScanDescriptor:
    points: int
    coordinate_dimensions: int
    coordinate_names: tuple[str, ...]


@dataclass(frozen=True)
class CaseOverride:
    name: str
    source_value: str
    canonical_value: str
    reason: str


@dataclass(frozen=True)
class ValidationStatus:
    input: str
    initialization: str
    matrix: str
    sampled_spectrum: str
    full_scan: str
    backend: str
    paper_target: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class BOCaseRecord:
    case_id: str
    source_family: SourceFamily
    input_capsule: str
    model: ModelDescriptor
    scan: ScanDescriptor
    matrix_size: int
    roots_per_scan: int
    supported_backends: tuple[str, ...]
    default_backend: str
    execution_status: str
    reference_compatible: bool
    overrides: tuple[CaseOverride, ...]
    validation: ValidationStatus

    def __post_init__(self) -> None:
        if self.matrix_size <= 0 or self.roots_per_scan <= 0 or self.scan.points <= 0:
            raise ValueError(f"Invalid dimensions for {self.case_id}")
        if self.default_backend not in self.supported_backends:
            raise ValueError(f"Default backend is not supported for {self.case_id}")
        if self.execution_status == "data-only" and self.supported_backends:
            raise ValueError(f"A data-only case cannot advertise solver backends: {self.case_id}")

    @property
    def solver_compatible(self) -> bool:
        return self.execution_status == "source-executable" and bool(self.supported_backends)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["solver_compatible"] = self.solver_compatible
        return result


_LEGACY_EVIDENCE = (
    "reports/legacy_pdrk_python/coefficient_matrix_sentinel_parity.json",
    "reports/legacy_pdrk_python/minimal_full_scans/full_scan_parity_and_performance.json",
)


def _legacy_validation(case_id: str) -> ValidationStatus:
    full_scan = "passed-with-near-zero-cancellation-diagnostic" if case_id == "bo2018_dispersionsurface" else "passed"
    return ValidationStatus(
        input="passed-hash-locked",
        initialization="passed",
        matrix="passed-matlab-sentinel-parity",
        sampled_spectrum="passed-one-to-one",
        full_scan=full_scan,
        backend="minimal-root-count-closed-and-full-scan-matched",
        paper_target="labels-available-evaluation-only",
        evidence=_LEGACY_EVIDENCE
        + (f"reports/reference_root_alignment/{case_id}/manifest.json",),
    )


def _legacy_record(
    case_id: str,
    *,
    iem: int,
    n_harmonics: int,
    j_poles: int,
    jpole_variant: str,
    scan_points: int,
    coordinate_dimensions: int,
    matrix_size: int,
    overrides: tuple[CaseOverride, ...] = (),
) -> BOCaseRecord:
    return BOCaseRecord(
        case_id=case_id,
        source_family="2018_bo_pdrk",
        input_capsule=f"legacy_cases/{case_id}",
        model=ModelDescriptor(
            name="legacy_es3d" if iem == 0 else "legacy_em3d",
            iem=iem,
            n_harmonics=n_harmonics,
            j_poles=j_poles,
            jpole_variant=jpole_variant,
        ),
        scan=ScanDescriptor(
            points=scan_points,
            coordinate_dimensions=coordinate_dimensions,
            coordinate_names=("pa",) if coordinate_dimensions == 1 else ("pa", "pb"),
        ),
        matrix_size=matrix_size,
        roots_per_scan=matrix_size,
        supported_backends=("full", "minimal"),
        default_backend="minimal",
        execution_status="source-executable",
        reference_compatible=True,
        overrides=overrides,
        validation=_legacy_validation(case_id),
    )


_CASES = (
    _legacy_record(
        "bo2018_beammode",
        iem=1,
        n_harmonics=1,
        j_poles=8,
        jpole_variant="j8_i8_opt2",
        scan_points=100,
        coordinate_dimensions=1,
        matrix_size=297,
        overrides=(
            CaseOverride(
                "jpole_variant",
                "j8_i10_opt3",
                "j8_i8_opt2",
                "Historical labeled root pool predates the current v190307 default.",
            ),
        ),
    ),
    _legacy_record(
        "bo2018_coldplasma",
        iem=1,
        n_harmonics=1,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=199,
        coordinate_dimensions=1,
        matrix_size=153,
    ),
    _legacy_record(
        "bo2018_darwin",
        iem=1,
        n_harmonics=1,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=250,
        coordinate_dimensions=1,
        matrix_size=153,
        overrides=(CaseOverride("iem", "2", "1", "The canonical 13-case dataset uses the EM panel."),),
    ),
    _legacy_record(
        "bo2018_dispersionsurface",
        iem=1,
        n_harmonics=3,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=969,
        coordinate_dimensions=2,
        matrix_size=345,
    ),
    _legacy_record(
        "bo2018_es1d_beam",
        iem=0,
        n_harmonics=1,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=50,
        coordinate_dimensions=1,
        matrix_size=49,
    ),
    _legacy_record(
        "bo2018_es3d_losscone",
        iem=0,
        n_harmonics=4,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=191,
        coordinate_dimensions=1,
        matrix_size=145,
    ),
    _legacy_record(
        "bo2018_loss_cone_mirror",
        iem=1,
        n_harmonics=3,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=100,
        coordinate_dimensions=1,
        matrix_size=345,
    ),
    _legacy_record(
        "bo2018_muschietti17",
        iem=1,
        n_harmonics=60,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=131,
        coordinate_dimensions=1,
        matrix_size=2961,
        overrides=(CaseOverride("n_harmonics", "3", "60", "Canonical high-dimensional dataset override."),),
    ),
    _legacy_record(
        "bo2018_ringbeam_umeda12",
        iem=1,
        n_harmonics=8,
        j_poles=8,
        jpole_variant="j8_i10_opt3",
        scan_points=117,
        coordinate_dimensions=1,
        matrix_size=825,
    ),
    BOCaseRecord(
        case_id="bo2025_firehose_astfalk17_theta45",
        source_family="2025_bo_arbitrary_hh",
        input_capsule="paper_cases/firehose_astfalk17_theta45",
        model=ModelDescriptor("hh_em3d", 1, 2, 24),
        scan=ScanDescriptor(46, 1, ("k",)),
        matrix_size=732,
        roots_per_scan=732,
        supported_backends=("full", "minimal"),
        default_backend="full",
        execution_status="source-executable",
        reference_compatible=True,
        overrides=(),
        validation=ValidationStatus(
            "passed-hash-locked",
            "passed",
            "passed-hh-tensor-tests",
            "passed-with-cross-solver-tolerance",
            "python-full-scan-available",
            "full-and-minimal; 46-point count audit passed; scans 0/1 rank-ambiguous diagnostic",
            "aligned-max-normalized-error-1.80e-4",
            (
                "reports/reference_root_alignment/bo2025_firehose_astfalk17_theta45/manifest.json",
                "reports/bo_fast_cpc/hh_minimal_firehose/firehose_46_full_vs_minimal.json",
            ),
        ),
    ),
    BOCaseRecord(
        case_id="bo2025_ice_irvine18",
        source_family="2025_bo_arbitrary_hh",
        input_capsule="paper_cases/ice_irvine18_ice",
        model=ModelDescriptor("hh_em3d", 1, 22, 4),
        scan=ScanDescriptor(781, 1, ("k",)),
        matrix_size=1635,
        roots_per_scan=1635,
        supported_backends=("full", "minimal"),
        default_backend="full",
        execution_status="source-executable",
        reference_compatible=False,
        overrides=(
            CaseOverride(
                "source_variant",
                "raw N=12, k=9.5..11.5",
                "archive_replay N=22, k=0.5..20.0",
                "The archived all-root panel records a different setup than raw bo_setup.m.",
            ),
        ),
        validation=ValidationStatus(
            "passed-hash-locked",
            "passed-archive-replay",
            "passed-hh-tensor-tests",
            "reference-source-mismatch-diagnosed",
            "not-accepted-against-archive",
            "full-and-minimal-implemented; case-wide minimal audit pending",
            "wpdat-labels-available-evaluation-only",
            (
                "paper_runs/ice_irvine18_ice/ice_source_diagnostics_sampled-12-of-781.json",
                "reports/reference_root_alignment/bo2025_ice_irvine18/manifest.json",
            ),
        ),
    ),
    BOCaseRecord(
        case_id="bo2025_ringbeam_umeda12_theta40",
        source_family="2025_bo_arbitrary_hh",
        input_capsule="paper_cases/ringbeam_umeda12_theta40",
        model=ModelDescriptor("hh_em3d", 1, 6, 12),
        scan=ScanDescriptor(117, 1, ("k",)),
        matrix_size=948,
        roots_per_scan=948,
        supported_backends=("full", "minimal"),
        default_backend="full",
        execution_status="source-executable",
        reference_compatible=True,
        overrides=(),
        validation=ValidationStatus(
            "passed-hash-locked",
            "passed",
            "passed-same-coefficient-sentinel",
            "passed-paper-branch-parity",
            "passed-paper-branch-full-scan",
            "full-and-minimal-implemented; case-wide minimal audit pending",
            "aligned-with-approximate-labels-declared",
            (
                "paper_runs/ringbeam_umeda12_theta40/ringbeam_hh_archive_parity_full-scan.json",
                "reports/reference_root_alignment/bo2025_ringbeam_umeda12_theta40/manifest.json",
            ),
        ),
    ),
    BOCaseRecord(
        case_id="bo2025_shell_min15_theta89",
        source_family="2025_bo_arbitrary_hh",
        input_capsule="paper_cases/shell_min15_theta89",
        model=ModelDescriptor("hh_em3d", 1, 10, 24),
        scan=ScanDescriptor(111, 1, ("k",)),
        matrix_size=4551,
        roots_per_scan=4551,
        supported_backends=("full", "minimal"),
        default_backend="full",
        execution_status="source-executable",
        reference_compatible=True,
        overrides=(),
        validation=ValidationStatus(
            "passed-hash-locked",
            "passed",
            "passed-same-coefficient-sentinel-mixed-provenance",
            "passed-paper-branch-parity",
            "passed-paper-branch-full-scan",
            "full-and-minimal-implemented; case-wide minimal audit pending",
            "aligned-max-normalized-error-4.80e-3",
            (
                "paper_runs/shell_min15_theta89/shell_hh_archive_parity_full-scan.json",
                "reports/reference_root_alignment/bo2025_shell_min15_theta89/manifest.json",
            ),
        ),
    ),
)


CASE_REGISTRY = {record.case_id: record for record in _CASES}
if len(CASE_REGISTRY) != 13:
    raise AssertionError("The canonical BO registry must contain exactly 13 unique cases")


def list_case_records() -> tuple[BOCaseRecord, ...]:
    return tuple(CASE_REGISTRY[case_id] for case_id in sorted(CASE_REGISTRY))


def get_case_record(case_id: str) -> BOCaseRecord:
    try:
        return CASE_REGISTRY[case_id]
    except KeyError as exc:
        choices = ", ".join(sorted(CASE_REGISTRY))
        raise KeyError(f"Unknown BO case {case_id!r}; choices: {choices}") from exc

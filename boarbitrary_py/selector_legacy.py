from __future__ import annotations

"""Label-blind local-count selector for the 2018 legacy BO/PDRK family.

The prediction path consumes one source spectrum and ordinary case inputs.
Alternative full spectra, paper curves, branch tables, and evaluation labels
are intentionally absent from this module.  The small local operators are the
family-correct scalar ES or 3x3 EM/generalized-Darwin Schur complements.
"""

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .case_registry import BOCaseRecord, get_case_record
from .hh_adaptive_contour_selector import (
    RootCountDecision,
    TierSelectionResult as NumericalTierSelectionResult,
    extend_tier_selection,
    fast_adjacent_prefilter,
    select_roots_by_local_counts,
)
from .legacy_pdrk.config import LegacyCaseInputs, LegacyContext, LegacyModelConfig
from .legacy_pdrk.analytic_rank import (
    LEGACY_MAGNETIZED_SINGLETON_RANK2_DEFECT_TOLERANCE,
    certify_magnetized_singleton_rank2,
)
from .legacy_pdrk.initialize import initialize_legacy_case
from .legacy_pdrk.io import load_legacy_case
from .legacy_pdrk.local_operator import (
    LegacyEMRationalOperator,
    LegacyESRationalOperator,
    LegacyRationalOperator,
    build_legacy_local_operator,
)
from .legacy_pdrk.scan import iter_legacy_scan_points
from .selector_artifacts import (
    FrozenDiagnosticPaths,
    FrozenPredictionPaths,
    freeze_prediction_artifact,
    freeze_selection_diagnostic,
)
from .selector_models import (
    ContourLevelEvidence,
    DeclaredPole,
    DiscretizationMetadata,
    PredictionManifest,
    ProbeEvidence,
    RootSelectionEvidence,
    SelectorFinalState,
    SelectorProfile,
    SelectorTiming,
    SourceSpectrumProvenance,
    SpectrumAccounting,
    StructuralPoleEvidence,
    TierSelectionResult,
)
from .selector_profiles import get_selector_profile
from .unified_runner import run_registered_case


LEGACY_FAMILY = "2018_bo_pdrk"
LegacyDiscretization = tuple[int, int, str]
LEGACY_STRUCTURAL_POLE_POLICY_SVD_STRICT_V1 = "svd_strict_v1"
LEGACY_STRUCTURAL_POLE_POLICY_MAGNETIZED_SINGLETON_RANK2_FLOOR_V1 = (
    "magnetized_singleton_rank2_floor_v1"
)
_LEGACY_STRUCTURAL_POLE_POLICIES = frozenset(
    {
        LEGACY_STRUCTURAL_POLE_POLICY_SVD_STRICT_V1,
        LEGACY_STRUCTURAL_POLE_POLICY_MAGNETIZED_SINGLETON_RANK2_FLOOR_V1,
    }
)
_ANALYTIC_RANK2_PROVENANCE = (
    "analytic magnetized-singleton rank<=2 certificate; collisionless "
    "coefficient formula; normalized B=L@R defect verified"
)


@dataclass(frozen=True)
class LegacyPoleMatchRecord:
    raw_index: int
    declared_pole: DeclaredPole
    multiplicity_slot: int
    normalized_distance: float


@dataclass(frozen=True)
class LegacyPoleRemoval:
    retained_roots: np.ndarray
    retained_indices: np.ndarray
    removed_indices: np.ndarray
    ambiguous_raw_indices: np.ndarray
    matches: tuple[LegacyPoleMatchRecord, ...]
    match_tolerance: float | None
    max_normalized_match_distance: float | None
    status: str
    message: str
    certified_occurrence_count: int
    ambiguous_occurrence_count: int
    ambiguous_proximity_guard_tolerance: float
    certified_poles_sha256: str
    structural_pole_policy: str = LEGACY_STRUCTURAL_POLE_POLICY_SVD_STRICT_V1
    analytic_certified_occurrence_count: int = 0
    analytic_certified_group_count: int = 0
    analytic_certificate_max_relative_factor_defect: float | None = None
    remaining_ambiguous_group_count: int = 0


@dataclass(frozen=True)
class LegacySelectionRunResult:
    manifest: PredictionManifest
    paths: FrozenPredictionPaths
    source_spectrum_origin: str
    all_root_audit_disagreement_count: int | None


class LegacySelectionAbstained(RuntimeError):
    """Raised after a whole-scan legacy diagnostic has been committed."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: Mapping[str, Any],
        paths: FrozenDiagnosticPaths,
    ) -> None:
        super().__init__(message)
        self.diagnostic = dict(diagnostic)
        self.paths = paths


def select_legacy_case(
    case_id: str,
    *,
    scan_index: int,
    output_dir: str | Path,
    workspace: str | Path | None = None,
    source_spectrum_path: str | Path | None = None,
    profile: SelectorProfile | None = None,
    stability_tolerance: float = 1.0e-3,
    quadrature_order: int = 120,
    include_core: bool = True,
    audit_all_roots: bool = False,
    artifact_stem: str | None = None,
    structural_pole_policy: str = LEGACY_STRUCTURAL_POLE_POLICY_SVD_STRICT_V1,
) -> LegacySelectionRunResult:
    """Classify one legacy spectrum using only local alternative operators.

    If ``source_spectrum_path`` is absent, exactly one registered full-pencil
    solve is performed.  A supplied artifact must expose exactly one normalized
    spectrum row, making its bytes and SHA-256 an explicit selector input.
    """

    if structural_pole_policy not in _LEGACY_STRUCTURAL_POLE_POLICIES:
        allowed = ", ".join(sorted(_LEGACY_STRUCTURAL_POLE_POLICIES))
        raise ValueError(
            f"unknown legacy structural-pole policy {structural_pole_policy!r}; "
            f"expected one of: {allowed}"
        )

    record = get_case_record(case_id)
    _require_legacy_record(record, scan_index)
    selected_profile = profile or get_selector_profile(case_id)
    if profile is None and not math.isclose(
        stability_tolerance,
        selected_profile.tolerance.normalized_movement_epsilon,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        selected_profile = replace(
            selected_profile,
            tolerance=replace(
                selected_profile.tolerance,
                normalized_movement_epsilon=stability_tolerance,
            ),
            metadata={
                **dict(selected_profile.metadata),
                "explicit_tolerance_override": True,
            },
        )
    _validate_legacy_profile(selected_profile, record, scan_index)
    if selected_profile.fallback is None:
        raise ValueError("the legacy fast selector requires one declared fallback probe")

    workspace_root = (
        Path(workspace).resolve()
        if workspace is not None
        else Path(__file__).resolve().parents[1]
    )
    source_started = perf_counter()
    if source_spectrum_path is None:
        source_result = run_registered_case(
            case_id,
            workspace=workspace_root,
            scan_indices=[scan_index],
            backend="full",
            quadrature_order=quadrature_order,
        )
        source_roots = np.asarray(source_result.normalized_roots[0], dtype=np.complex128)
        source_internal_seconds = float(source_result.elapsed_seconds)
        source_origin = "single_registered_full_solve"
        source_artifact = ""
        source_artifact_hash = None
    else:
        source_path = Path(source_spectrum_path).resolve()
        source_roots = _load_one_source_spectrum(source_path, scan_index=scan_index)
        source_internal_seconds = 0.0
        source_origin = "single_frozen_source_spectrum"
        source_artifact = str(source_path)
        source_artifact_hash = _sha256_path(source_path)
    source_wall_seconds = perf_counter() - source_started
    if source_roots.shape != (record.roots_per_scan,):
        raise ValueError(
            f"source spectrum has {source_roots.size} roots; registry requires "
            f"{record.roots_per_scan}"
        )
    finite_mask = np.isfinite(source_roots.real) & np.isfinite(source_roots.imag)
    if not np.all(finite_mask):
        diagnostic = {
            "schema_version": 1,
            "status": "unsupported",
            "reason": "nonfinite_generalized_eigenvalues_not_representable_in_manifest_v1",
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "case_id": case_id,
            "scan_index_0based": scan_index,
            "registered_iem": int(record.model.iem),
            "source_root_count": int(source_roots.size),
            "finite_root_count": int(np.count_nonzero(finite_mask)),
            "nonfinite_root_count": int(np.count_nonzero(~finite_mask)),
            "nonfinite_raw_indices_0based": [
                int(value) for value in np.flatnonzero(~finite_mask)
            ],
            "source_spectrum_sha256": _spectrum_sha256_allow_nonfinite(source_roots),
            "source_spectrum_origin": source_origin,
            "profile_id": selected_profile.profile_id,
            "prediction_manifest_written": False,
            "scientific_interpretation": (
                "The generalized pencil produced infinite/NaN constraint eigenvalues. "
                "They were retained in this diagnostic instead of being dropped or "
                "misreported as finite BO-stable roots."
            ),
        }
        stem = artifact_stem or (
            f"{case_id}__scan-{scan_index:04d}__selection-abstention"
        )
        paths = freeze_selection_diagnostic(diagnostic, output_dir, stem=stem)
        raise LegacySelectionAbstained(
            diagnostic["reason"], diagnostic=diagnostic, paths=paths
        )

    source_config = _legacy_config(selected_profile.source.discretization)
    primary_configs = tuple(
        _legacy_config(item) for item in selected_profile.near.discretizations
    )
    fallback_config = _legacy_config(selected_profile.fallback.discretization)
    core_configs = (
        ()
        if not include_core or selected_profile.core is None
        else tuple(_legacy_config(item) for item in selected_profile.core.discretizations)
    )
    all_configs = tuple(
        dict.fromkeys((source_config, *primary_configs, fallback_config, *core_configs))
    )

    operator_started = perf_counter()
    inputs = load_legacy_case(case_id, cases_root=workspace_root / "legacy_cases")
    operators = _build_legacy_operators(
        inputs,
        scan_index=scan_index,
        configs=all_configs,
        quadrature_order=quadrature_order,
    )
    operator_build_seconds = perf_counter() - operator_started
    source_operator = operators[source_config]
    pole_removal = _remove_certified_legacy_poles(
        source_roots,
        source_operator,
        structural_pole_policy=structural_pole_policy,
    )
    if pole_removal.match_tolerance is not None:
        selected_profile = replace(
            selected_profile,
            tolerance=replace(
                selected_profile.tolerance,
                structural_pole_match_tolerance=pole_removal.match_tolerance,
            ),
        )

    retained_roots = pole_removal.retained_roots
    retained_indices = pole_removal.retained_indices
    tolerance = selected_profile.tolerance
    contour = selected_profile.contour
    near_operator_configs = tuple(
        dict.fromkeys((source_config, fallback_config, *primary_configs))
    )
    near_operators = {
        config: operators[config] for config in near_operator_configs
    }
    prefilter = fast_adjacent_prefilter(
        retained_roots,
        retained_indices,
        near_operators,
        source_discretization=source_config,
        primary_probes=primary_configs,
        fallback_probe=fallback_config,
        stability_tolerance=tolerance.normalized_movement_epsilon,
        root_separation_fraction=tolerance.root_separation_fraction,
        pole_separation_fraction=tolerance.pole_separation_fraction,
        tolerance_radius_fraction=tolerance.tolerance_radius_fraction,
        zero_tolerance=tolerance.zero_root_tolerance,
        initial_samples=contour.sample_levels[0],
        maximum_samples=contour.sample_levels[-1],
        maximum_phase_increment=contour.maximum_phase_increment,
    )
    near_result = select_roots_by_local_counts(
        retained_roots,
        retained_indices,
        near_operators,
        source_discretization=source_config,
        tier_name="near_stable",
        stability_tolerance=tolerance.normalized_movement_epsilon,
        root_separation_fraction=tolerance.root_separation_fraction,
        pole_separation_fraction=tolerance.pole_separation_fraction,
        tolerance_radius_fraction=tolerance.tolerance_radius_fraction,
        zero_tolerance=tolerance.zero_root_tolerance,
        initial_samples=contour.sample_levels[0],
        maximum_samples=contour.sample_levels[-1],
        maximum_phase_increment=contour.maximum_phase_increment,
        precomputed_centers=prefilter.source_centers,
        geometry_operators=tuple(operators.values()),
        active_positions=prefilter.selected_positions,
    )
    if core_configs:
        core_result = extend_tier_selection(
            near_result,
            {config: operators[config] for config in core_configs},
            tier_name="core_stable",
            initial_samples=contour.sample_levels[0],
            maximum_samples=contour.sample_levels[-1],
            maximum_phase_increment=contour.maximum_phase_increment,
        )
    else:
        core_result = None

    audit_result = None
    audit_disagreement = None
    if audit_all_roots:
        audit_result = select_roots_by_local_counts(
            retained_roots,
            retained_indices,
            near_operators,
            source_discretization=source_config,
            tier_name="all_root_near_audit",
            stability_tolerance=tolerance.normalized_movement_epsilon,
            root_separation_fraction=tolerance.root_separation_fraction,
            pole_separation_fraction=tolerance.pole_separation_fraction,
            tolerance_radius_fraction=tolerance.tolerance_radius_fraction,
            zero_tolerance=tolerance.zero_root_tolerance,
            initial_samples=contour.sample_levels[0],
            maximum_samples=contour.sample_levels[-1],
            maximum_phase_increment=contour.maximum_phase_increment,
            precomputed_centers=prefilter.source_centers,
            geometry_operators=tuple(operators.values()),
        )
        audit_disagreement = len(
            set(int(value) for value in near_result.selected_original_indices)
            ^ set(int(value) for value in audit_result.selected_original_indices)
        )

    roots = _build_root_evidence(
        source_roots,
        pole_removal,
        selected_profile,
        prefilter_status=prefilter.status_by_position,
        source_centers=prefilter.source_centers,
        near_result=near_result,
        core_result=core_result,
        audit_result=audit_result,
    )
    accounting = SpectrumAccounting.from_evidence(source_roots.size, roots)
    tier_results = [
        _tier_artifact(near_result, selected_profile.near.discretizations)
    ]
    if core_result is not None and selected_profile.core is not None:
        tier_results.append(
            _tier_artifact(core_result, selected_profile.core.discretizations)
        )
    if audit_result is not None:
        tier_results.append(
            _tier_artifact(
                audit_result,
                tuple(
                    _metadata_for_profile_config(selected_profile, config)
                    for config in near_operator_configs
                ),
            )
        )

    source_hash = _spectrum_sha256(source_roots)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    selector_elapsed_without_serialization = (
        operator_build_seconds
        + prefilter.elapsed_seconds
        + near_result.elapsed_seconds
        + (0.0 if core_result is None else core_result.elapsed_seconds)
    )
    provisional_timing = SelectorTiming(
        source_eigensolve_seconds=source_internal_seconds,
        operator_build_seconds=operator_build_seconds,
        prefilter_seconds=prefilter.elapsed_seconds,
        near_verification_seconds=near_result.elapsed_seconds,
        core_extension_seconds=0.0 if core_result is None else core_result.elapsed_seconds,
        prediction_serialization_seconds=0.0,
        selector_total_seconds=selector_elapsed_without_serialization,
        end_to_end_seconds=source_wall_seconds + selector_elapsed_without_serialization,
        metadata={"quadrature_order": quadrature_order},
    )
    adapter, operator_semantics = _adapter_metadata(record, source_operator)
    experimental_rank2_floor = (
        structural_pole_policy
        == LEGACY_STRUCTURAL_POLE_POLICY_MAGNETIZED_SINGLETON_RANK2_FLOOR_V1
    )
    family_metadata: dict[str, Any] = {
        "adapter": adapter,
        "operator_semantics": operator_semantics,
        "registered_iem": int(record.model.iem),
        "case_name_contains_darwin": case_id == "bo2018_darwin",
        "canonical_iem_override_applied": bool(
            case_id == "bo2018_darwin" and record.model.iem == 1
        ),
        "pole_certificate_kind": "legacy_mcmillan_residue_rank_v1",
        "pole_certificate_count_closure": bool(
            source_operator.mcmillan_certificate.count_closure
        ),
        "pole_certificate_ambiguous_rank_groups": int(
            source_operator.mcmillan_certificate.ambiguous_rank_count
        ),
        "certified_structural_occurrence_count": int(
            pole_removal.certified_occurrence_count
        ),
        "ambiguous_structural_occurrence_count": int(
            pole_removal.ambiguous_occurrence_count
        ),
        "ambiguous_structural_nearby_root_count": int(
            pole_removal.ambiguous_raw_indices.size
        ),
        "ambiguous_pole_proximity_guard_tolerance": (
            pole_removal.ambiguous_proximity_guard_tolerance
        ),
        "pole_match_status": pole_removal.status,
        "pole_match_message": pole_removal.message,
        "pole_match_tolerance": pole_removal.match_tolerance,
        "pole_match_max_normalized_distance": (
            pole_removal.max_normalized_match_distance
        ),
        "pole_certificate_sha256": pole_removal.certified_poles_sha256,
        "raw_pole_array_used_as_deletion_list": False,
        "all_root_audit_requested": audit_all_roots,
        "all_root_audit_disagreement_count": audit_disagreement,
        "scientific_label": "bo_discretization_stable_candidate_only",
    }
    if experimental_rank2_floor:
        family_metadata.update(
            {
                "structural_pole_policy": structural_pole_policy,
                "pole_certificate_kind": (
                    "legacy_mcmillan_residue_rank_v1_plus_"
                    "magnetized_singleton_rank2_floor_v1"
                ),
                "analytic_rank2_certified_occurrence_count": int(
                    pole_removal.analytic_certified_occurrence_count
                ),
                "analytic_rank2_certified_group_count": int(
                    pole_removal.analytic_certified_group_count
                ),
                "analytic_rank2_max_relative_factor_defect": (
                    pole_removal.analytic_certificate_max_relative_factor_defect
                ),
                "analytic_rank2_factor_defect_tolerance": (
                    LEGACY_MAGNETIZED_SINGLETON_RANK2_DEFECT_TOLERANCE
                ),
                "remaining_ambiguous_structural_group_count": int(
                    pole_removal.remaining_ambiguous_group_count
                ),
                "analytic_rank2_scope": (
                    "singleton magnetized EM/Darwin group; exact nu=0; "
                    "finite residue and pole; verified B=L@R defect"
                ),
            }
        )
    manifest = PredictionManifest(
        selector_id=(
            "bo-fast-legacy-local-count-forward-j-v2-exp-magnetized-rank2-floor-v1"
            if experimental_rank2_floor
            else "bo-fast-legacy-local-count-forward-j-v2"
        ),
        generated_at_utc=generated_at,
        source=SourceSpectrumProvenance(
            case_id=case_id,
            scan_index=scan_index,
            root_count=source_roots.size,
            spectrum_sha256=source_hash,
            solver_id=(
                "boarbitrary_py.unified_runner:legacy_full"
                if source_spectrum_path is None
                else "caller_supplied_single_normalized_npz"
            ),
            artifact_path=source_artifact,
            metadata={
                "source_origin": source_origin,
                "source_solve_wall_seconds": source_wall_seconds,
                "source_artifact_sha256": source_artifact_hash,
            },
        ),
        profile=selected_profile,
        roots=roots,
        tier_results=tuple(tier_results),
        accounting=accounting,
        timings=provisional_timing,
        numerical_failure_counts=_numerical_failure_counts(roots),
        family_metadata=family_metadata,
        selector_input_manifest_sha256=None,
    )
    serialization_started = perf_counter()
    manifest.canonical_json()
    serialization_seconds = perf_counter() - serialization_started
    final_timing = replace(
        provisional_timing,
        prediction_serialization_seconds=serialization_seconds,
        selector_total_seconds=selector_elapsed_without_serialization
        + serialization_seconds,
        end_to_end_seconds=(
            source_wall_seconds
            + selector_elapsed_without_serialization
            + serialization_seconds
        ),
    )
    manifest = replace(manifest, timings=final_timing)
    paths = freeze_prediction_artifact(manifest, output_dir, stem=artifact_stem)
    return LegacySelectionRunResult(
        manifest=manifest,
        paths=paths,
        source_spectrum_origin=source_origin,
        all_root_audit_disagreement_count=audit_disagreement,
    )


def _require_legacy_record(record: BOCaseRecord, scan_index: int) -> None:
    if record.source_family != LEGACY_FAMILY:
        raise ValueError(
            f"{record.case_id} is {record.source_family}; the legacy PDRK "
            "adapter cannot be used for that family"
        )
    if not 0 <= scan_index < record.scan.points:
        raise ValueError(
            f"scan_index {scan_index} is outside [0, {record.scan.points - 1}]"
        )


def _validate_legacy_profile(
    profile: SelectorProfile,
    record: BOCaseRecord,
    scan_index: int,
) -> None:
    if profile.family != LEGACY_FAMILY or profile.family != record.source_family:
        raise ValueError("selector profile family does not match the registered legacy case")
    if profile.scan_scope.case_id != record.case_id or not profile.scan_scope.contains(
        scan_index
    ):
        raise ValueError("selector profile scan scope does not admit the requested point")
    source = _legacy_config(profile.source.discretization)
    expected = (
        int(record.model.j_poles),
        int(record.model.n_harmonics),
        str(record.model.jpole_variant),
    )
    if source != expected:
        raise ValueError(
            f"selector source discretization {source} differs from registered {expected}"
        )
    refinement_probes = [*profile.near.discretizations]
    if profile.fallback is not None:
        refinement_probes.append(profile.fallback.discretization)
    if profile.core is not None:
        refinement_probes.extend(profile.core.discretizations)
    if any(int(probe.pole_order) < source[0] for probe in refinement_probes):
        raise ValueError(
            "legacy forward-J selector cannot use a lower-J probe as an acceptance gate"
        )


def _build_legacy_operators(
    base_inputs: LegacyCaseInputs,
    *,
    scan_index: int,
    configs: tuple[LegacyDiscretization, ...],
    quadrature_order: int,
) -> dict[LegacyDiscretization, LegacyRationalOperator]:
    operators: dict[LegacyDiscretization, LegacyRationalOperator] = {}
    reference_normalization: float | None = None
    for config in configs:
        probe_inputs = _inputs_for_discretization(base_inputs, config)
        ctx = initialize_legacy_case(probe_inputs)
        point = _scan_point(ctx, scan_index)
        operator = build_legacy_local_operator(
            ctx,
            point,
            quadrature_order=quadrature_order,
        )
        if reference_normalization is None:
            reference_normalization = operator.normalization.scale
        elif not math.isclose(
            operator.normalization.scale,
            reference_normalization,
            rel_tol=1.0e-13,
            abs_tol=0.0,
        ):
            raise ValueError("legacy probe normalization changed with discretization")
        actual = (
            int(operator.discretization.pole_order),
            int(operator.discretization.harmonic_truncation),
            str(operator.discretization.variant),
        )
        if actual != config:
            raise AssertionError(
                f"legacy local operator metadata {actual} differs from requested {config}"
            )
        operators[config] = operator
    return operators


def _inputs_for_discretization(
    inputs: LegacyCaseInputs,
    config: LegacyDiscretization,
) -> LegacyCaseInputs:
    j_poles, n_harmonics, variant = config
    model = replace(
        inputs.config.model,
        j_poles=int(j_poles),
        n_harmonics=int(n_harmonics),
        jpole_variant=str(variant),
    )
    matrix_size = _matrix_size_for_model(inputs, model)
    canonical = replace(
        inputs.config.canonical,
        matrix_size=matrix_size,
        roots_per_scan=matrix_size,
    )
    config_value = replace(inputs.config, model=model, canonical=canonical)
    return replace(inputs, config=config_value)


def _matrix_size_for_model(
    inputs: LegacyCaseInputs,
    model: LegacyModelConfig,
) -> int:
    species = inputs.species
    magnetized = np.asarray(
        [item.magnetized for item in species], dtype=bool
    )
    if inputs.config.b0 == 0.0:
        magnetized[:] = False
    ns = np.full(len(species), 2 * model.n_harmonics + 1, dtype=np.int64)
    for row_index, (declared_species, n_value) in enumerate(model.set_ns):
        if declared_species <= len(species) and row_index < len(species):
            ns[row_index] = 2 * int(n_value) + 1
    for index, item in enumerate(species):
        if not magnetized[index]:
            ns[index] = 1 if item.alpha == 1.0 or item.delta == 1.0 else 2
    snj = int(np.sum(ns) * model.j_poles)
    return snj + 1 if model.iem == 0 else 3 * (snj + 1) + 6


def _scan_point(ctx: LegacyContext, scan_index: int):
    for point in iter_legacy_scan_points(ctx):
        if point.flat_index == scan_index:
            return point
    raise IndexError(f"legacy scan index {scan_index} is unavailable")


def _remove_certified_legacy_poles(
    source_roots: np.ndarray,
    operator: LegacyRationalOperator,
    *,
    structural_pole_policy: str = LEGACY_STRUCTURAL_POLE_POLICY_SVD_STRICT_V1,
    match_tolerances: tuple[float, ...] = (
        1.0e-12,
        3.0e-12,
        1.0e-11,
        3.0e-11,
        1.0e-10,
        3.0e-10,
        1.0e-9,
        3.0e-9,
        1.0e-8,
        3.0e-8,
        1.0e-7,
        3.0e-7,
        1.0e-6,
        3.0e-6,
        1.0e-5,
        3.0e-5,
        1.0e-4,
    ),
) -> LegacyPoleRemoval:
    if structural_pole_policy not in _LEGACY_STRUCTURAL_POLE_POLICIES:
        allowed = ", ".join(sorted(_LEGACY_STRUCTURAL_POLE_POLICIES))
        raise ValueError(
            f"unknown legacy structural-pole policy {structural_pole_policy!r}; "
            f"expected one of: {allowed}"
        )
    roots = np.asarray(source_roots, dtype=np.complex128).reshape(-1)
    certificate = operator.mcmillan_certificate
    if certificate.original_size != roots.size:
        raise ValueError(
            "legacy source root count differs from the McMillan certificate"
        )

    declarations: list[tuple[DeclaredPole, int]] = []
    ambiguous_occurrences = 0
    ambiguous_values: list[complex] = []
    remaining_ambiguous_groups = 0
    analytic_certified_occurrences = 0
    analytic_certified_groups = 0
    analytic_factor_defects: list[float] = []
    for group_index, group in enumerate(operator.structural_groups):
        multiplicity = int(group.known_pole_multiplicity)
        if group.rank_ambiguous or not certificate.count_closure:
            analytic_floor = 0
            if (
                structural_pole_policy
                == LEGACY_STRUCTURAL_POLE_POLICY_MAGNETIZED_SINGLETON_RANK2_FLOOR_V1
                and certificate.count_closure
                and group.rank_ambiguous
                and int(group.block_size) == 3
                and np.asarray(group.coefficient_indices).size == 1
            ):
                coefficient_index = int(np.asarray(group.coefficient_indices)[0])
                analytic_certificate = certify_magnetized_singleton_rank2(
                    operator.ctx,
                    operator.point,
                    operator.coefficients,
                    coefficient_index,
                )
                if (
                    analytic_certificate is not None
                    and analytic_certificate.certified
                    and analytic_certificate.structural_multiplicity_floor == 1
                ):
                    analytic_floor = 1
                    analytic_factor_defects.append(
                        analytic_certificate.relative_factor_defect
                    )
                    analytic_certified_occurrences += 1
                    analytic_certified_groups += 1
                    declarations.append(
                        (
                            DeclaredPole(
                                value_normalized=complex(
                                    group.pole / operator.normalization.scale
                                ),
                                multiplicity=1,
                                provenance=_ANALYTIC_RANK2_PROVENANCE,
                                certificate_id=(
                                    f"{operator.ctx.case.case_id}:"
                                    f"scan-{operator.point.flat_index}:"
                                    f"group-{group_index}:coefficient-"
                                    f"{coefficient_index}:analytic-rank2-floor-v1:"
                                    f"defect-{analytic_certificate.relative_factor_defect:.3e}"
                                ),
                            ),
                            group_index,
                        )
                    )

            if multiplicity == 0 and analytic_floor == 0:
                continue
            ambiguous_occurrences += multiplicity
            ambiguous_occurrences -= min(analytic_floor, multiplicity)
            ambiguous_values.append(
                complex(group.pole / operator.normalization.scale)
            )
            remaining_ambiguous_groups += 1
            continue
        if multiplicity == 0:
            continue
        declared = DeclaredPole(
            value_normalized=complex(group.pole / operator.normalization.scale),
            multiplicity=multiplicity,
            provenance=(
                "LegacyMinimalGroup.known_pole_multiplicity; exact-bitwise pole "
                "grouping; non-ambiguous residue-rank certificate"
            ),
            certificate_id=(
                f"{operator.ctx.case.case_id}:scan-{operator.point.flat_index}:"
                f"group-{group_index}:rank-tolerance-"
                f"{certificate.rank_tolerance:.3e}"
            ),
        )
        declarations.append((declared, group_index))

    expanded = np.asarray(
        [
            declared.value_normalized
            for declared, _ in declarations
            for _slot in range(declared.multiplicity)
        ],
        dtype=np.complex128,
    )
    pole_hash = hashlib.sha256(
        np.ascontiguousarray(expanded, dtype="<c16").view(np.uint8)
    ).hexdigest()
    certified_count = int(expanded.size)
    if certified_count == 0:
        retained_indices = np.arange(roots.size, dtype=np.int64)
        return LegacyPoleRemoval(
            retained_roots=_readonly(roots),
            retained_indices=_readonly(retained_indices, dtype=np.int64),
            removed_indices=_readonly(np.empty(0, dtype=np.int64), dtype=np.int64),
            ambiguous_raw_indices=_readonly(
                _roots_near_values(
                    roots,
                    ambiguous_values,
                    tolerance=match_tolerances[-1],
                ),
                dtype=np.int64,
            ),
            matches=(),
            match_tolerance=None,
            max_normalized_match_distance=None,
            status=(
                "no_nonambiguous_structural_multiplicity"
                if ambiguous_occurrences
                else "no_structural_poles"
            ),
            message=(
                "Ambiguous structural multiplicities were retained as selector "
                "candidates and will abstain near declared pole geometry."
                if ambiguous_occurrences
                else "The source certificate declares no structural poles."
            ),
            certified_occurrence_count=0,
            ambiguous_occurrence_count=ambiguous_occurrences,
            ambiguous_proximity_guard_tolerance=float(match_tolerances[-1]),
            certified_poles_sha256=pole_hash,
            structural_pole_policy=structural_pole_policy,
            analytic_certified_occurrence_count=analytic_certified_occurrences,
            analytic_certified_group_count=analytic_certified_groups,
            analytic_certificate_max_relative_factor_defect=max(
                analytic_factor_defects, default=None
            ),
            remaining_ambiguous_group_count=remaining_ambiguous_groups,
        )

    chosen: tuple[LegacyPoleMatchRecord, ...] | None = None
    chosen_tolerance: float | None = None
    last_max: float | None = None
    for tolerance in match_tolerances:
        if tolerance <= 0.0:
            raise ValueError("pole-match tolerances must be positive")
        available = np.ones(roots.size, dtype=bool)
        attempt: list[LegacyPoleMatchRecord] = []
        failed = False
        for declared, _group_index in declarations:
            available_indices = np.flatnonzero(available)
            if available_indices.size < declared.multiplicity:
                failed = True
                break
            candidates = roots[available_indices]
            distances = np.abs(candidates - declared.value_normalized) / (
                1.0 + np.maximum(np.abs(candidates), abs(declared.value_normalized))
            )
            take = np.argpartition(distances, declared.multiplicity - 1)[
                : declared.multiplicity
            ]
            order = take[np.argsort(distances[take], kind="stable")]
            selected = available_indices[order]
            selected_distances = distances[order]
            last_max = float(np.max(selected_distances, initial=0.0))
            if np.any(selected_distances > tolerance):
                failed = True
                break
            for slot, (raw_index, distance) in enumerate(
                zip(selected, selected_distances, strict=True)
            ):
                attempt.append(
                    LegacyPoleMatchRecord(
                        raw_index=int(raw_index),
                        declared_pole=declared,
                        multiplicity_slot=slot,
                        normalized_distance=float(distance),
                    )
                )
            available[selected] = False
        if not failed and len(attempt) == certified_count:
            chosen = tuple(attempt)
            chosen_tolerance = float(tolerance)
            break

    if chosen is None or chosen_tolerance is None:
        retained_indices = np.arange(roots.size, dtype=np.int64)
        return LegacyPoleRemoval(
            retained_roots=_readonly(roots),
            retained_indices=_readonly(retained_indices, dtype=np.int64),
            removed_indices=_readonly(np.empty(0, dtype=np.int64), dtype=np.int64),
            ambiguous_raw_indices=_readonly(
                _roots_near_values(
                    roots,
                    [
                        *(item.value_normalized for item, _ in declarations),
                        *ambiguous_values,
                    ],
                    tolerance=match_tolerances[-1],
                ),
                dtype=np.int64,
            ),
            matches=(),
            match_tolerance=None,
            max_normalized_match_distance=last_max,
            status="certified_multiplicity_match_failed_roots_preserved",
            message=(
                "Certified pole occurrences could not be matched within the fixed "
                "guard sequence; all nearby roots were preserved for unresolved "
                "classification instead of heuristic deletion."
            ),
            certified_occurrence_count=certified_count,
            ambiguous_occurrence_count=ambiguous_occurrences,
            ambiguous_proximity_guard_tolerance=float(match_tolerances[-1]),
            certified_poles_sha256=pole_hash,
            structural_pole_policy=structural_pole_policy,
            analytic_certified_occurrence_count=analytic_certified_occurrences,
            analytic_certified_group_count=analytic_certified_groups,
            analytic_certificate_max_relative_factor_defect=max(
                analytic_factor_defects, default=None
            ),
            remaining_ambiguous_group_count=remaining_ambiguous_groups,
        )

    removed_indices = np.asarray(
        sorted(item.raw_index for item in chosen), dtype=np.int64
    )
    retained_mask = np.ones(roots.size, dtype=bool)
    retained_mask[removed_indices] = False
    retained_indices = np.flatnonzero(retained_mask).astype(np.int64)
    ambiguous_raw_indices = _roots_near_values(
        roots,
        ambiguous_values,
        tolerance=match_tolerances[-1],
        excluded=removed_indices,
    )
    max_distance = max(item.normalized_distance for item in chosen)
    return LegacyPoleRemoval(
        retained_roots=_readonly(roots[retained_indices]),
        retained_indices=_readonly(retained_indices, dtype=np.int64),
        removed_indices=_readonly(removed_indices, dtype=np.int64),
        ambiguous_raw_indices=_readonly(ambiguous_raw_indices, dtype=np.int64),
        matches=chosen,
        match_tolerance=chosen_tolerance,
        max_normalized_match_distance=float(max_distance),
        status=(
            "matched_mcmillan_and_analytic_rank2_floor_multiplicities"
            if analytic_certified_occurrences
            else "matched_nonambiguous_mcmillan_multiplicities"
        ),
        message=(
            "Non-ambiguous McMillan multiplicities and one analytically guaranteed "
            "occurrence per eligible singleton magnetized ambiguous group were "
            "removed; every such pole remains proximity-guarded for unresolved "
            "residual multiplicity."
            if analytic_certified_occurrences
            else "Only non-ambiguous McMillan structural multiplicities were removed; "
            "ambiguous groups remain in the source-root accounting."
        ),
        certified_occurrence_count=certified_count,
        ambiguous_occurrence_count=ambiguous_occurrences,
        ambiguous_proximity_guard_tolerance=float(match_tolerances[-1]),
        certified_poles_sha256=pole_hash,
        structural_pole_policy=structural_pole_policy,
        analytic_certified_occurrence_count=analytic_certified_occurrences,
        analytic_certified_group_count=analytic_certified_groups,
        analytic_certificate_max_relative_factor_defect=max(
            analytic_factor_defects, default=None
        ),
        remaining_ambiguous_group_count=remaining_ambiguous_groups,
    )


def _build_root_evidence(
    source_roots: np.ndarray,
    pole_removal: LegacyPoleRemoval,
    profile: SelectorProfile,
    *,
    prefilter_status: tuple[str, ...],
    source_centers: tuple[Any, ...],
    near_result: NumericalTierSelectionResult,
    core_result: NumericalTierSelectionResult | None,
    audit_result: NumericalTierSelectionResult | None,
) -> tuple[RootSelectionEvidence, ...]:
    pole_evidence: dict[int, StructuralPoleEvidence] = {}
    pole_stage_status: dict[int, str] = {}
    for item in pole_removal.matches:
        analytic_rank2_floor = item.declared_pole.provenance == _ANALYTIC_RANK2_PROVENANCE
        pole_message = (
            "matched to an analytic magnetized-singleton rank<=2 structural "
            "multiplicity floor; "
            f"match_guard={pole_removal.match_tolerance:.3e}; "
            "residual_ambiguity_guard="
            f"{pole_removal.ambiguous_proximity_guard_tolerance:.3e}"
            if analytic_rank2_floor
            else "matched to a non-ambiguous McMillan structural multiplicity; "
            f"guard={pole_removal.match_tolerance:.3e}"
        )
        pole_evidence[item.raw_index] = StructuralPoleEvidence(
            matched=True,
            declared_pole=item.declared_pole,
            matching_distance=item.normalized_distance,
            multiplicity_slot=item.multiplicity_slot,
            message=pole_message,
        )
        pole_stage_status[item.raw_index] = (
            "matched_analytic_magnetized_singleton_rank2_floor_certificate"
            if analytic_rank2_floor
            else "matched_nonambiguous_mcmillan_certificate"
        )
    retained_position = {
        int(raw_index): position
        for position, raw_index in enumerate(pole_removal.retained_indices)
    }
    near_selected = set(int(value) for value in near_result.selected_original_indices)
    core_selected = (
        set()
        if core_result is None
        else set(int(value) for value in core_result.selected_original_indices)
    )
    audit_selected = (
        None
        if audit_result is None
        else set(int(value) for value in audit_result.selected_original_indices)
    )
    near_by_position = {
        item.retained_position: item for item in near_result.decisions
    }
    core_by_position = (
        {}
        if core_result is None
        else {item.retained_position: item for item in core_result.decisions}
    )
    audit_by_position = (
        {}
        if audit_result is None
        else {item.retained_position: item for item in audit_result.decisions}
    )
    epsilon = profile.tolerance.normalized_movement_epsilon
    forced_unresolved = set(
        int(value) for value in pole_removal.ambiguous_raw_indices
    )
    rows: list[RootSelectionEvidence] = []
    for raw_index, root in enumerate(source_roots):
        if raw_index in pole_evidence:
            rows.append(
                RootSelectionEvidence(
                    raw_index=raw_index,
                    source_root=complex(root),
                    final_state=SelectorFinalState.DECLARED_AUXILIARY_POLE,
                    structural_pole=pole_evidence[raw_index],
                    stability_tolerance=epsilon,
                    stage_statuses={
                        "structural_pole": pole_stage_status[raw_index]
                    },
                )
            )
            continue

        position = retained_position[raw_index]
        near_decision = near_by_position[position]
        core_decision = core_by_position.get(position)
        audit_decision = audit_by_position.get(position)
        statuses = {
            "prefilter": str(prefilter_status[position]),
            "near": near_decision.status,
        }
        if core_decision is not None:
            statuses["core"] = core_decision.status
        if audit_decision is not None:
            statuses["all_root_audit"] = audit_decision.status
        if raw_index in forced_unresolved:
            statuses["structural_pole"] = "unresolved_ambiguous_multiplicity"
        state = _final_state(
            raw_index,
            statuses,
            near_selected=near_selected,
            core_selected=core_selected,
            audit_selected=audit_selected,
            forced_unresolved=raw_index in forced_unresolved,
        )
        decision = _best_decision(
            state,
            near_decision=near_decision,
            core_decision=core_decision,
            audit_decision=audit_decision,
        )
        probes = _probe_evidence(decision, profile)
        failed_probe = None
        if decision.first_failed_discretization is not None:
            candidate = _metadata_for_profile_config(
                profile, decision.first_failed_discretization
            )
            if candidate.key in {item.discretization.key for item in probes}:
                failed_probe = candidate
        center_info = source_centers[position]
        center = complex(center_info.center) if center_info.finite else None
        radius = (
            float(decision.radius)
            if math.isfinite(decision.radius) and decision.radius > 0.0
            else None
        )
        rows.append(
            RootSelectionEvidence(
                raw_index=raw_index,
                retained_position=position,
                source_root=complex(root),
                final_state=state,
                corrected_center=center,
                isolation_radius=radius,
                probes=probes,
                failed_probe=failed_probe,
                stability_tolerance=epsilon,
                stage_statuses=statuses,
            )
        )
    return tuple(rows)


def _final_state(
    raw_index: int,
    statuses: Mapping[str, str],
    *,
    near_selected: set[int],
    core_selected: set[int],
    audit_selected: set[int] | None,
    forced_unresolved: bool = False,
) -> SelectorFinalState:
    values = tuple(statuses.values())
    if any(value == "zero_or_constraint_root" for value in values):
        return SelectorFinalState.ZERO_OR_CONSTRAINT_ROOT
    if forced_unresolved:
        return SelectorFinalState.UNRESOLVED
    if raw_index in core_selected:
        return SelectorFinalState.CORE_STABLE
    if raw_index in near_selected:
        return SelectorFinalState.NEAR_STABLE_ONLY
    if audit_selected is not None and raw_index in audit_selected:
        return SelectorFinalState.UNRESOLVED
    if any(value.startswith("unresolved") for value in values):
        return SelectorFinalState.UNRESOLVED
    return SelectorFinalState.REJECTED


def _best_decision(
    state: SelectorFinalState,
    *,
    near_decision: RootCountDecision,
    core_decision: RootCountDecision | None,
    audit_decision: RootCountDecision | None,
) -> RootCountDecision:
    if state in {SelectorFinalState.CORE_STABLE, SelectorFinalState.NEAR_STABLE_ONLY}:
        return core_decision or near_decision
    return audit_decision or near_decision


def _probe_evidence(
    decision: RootCountDecision,
    profile: SelectorProfile,
) -> tuple[ProbeEvidence, ...]:
    rows: list[ProbeEvidence] = []
    for config, evidence in decision.contour_evidence:
        levels = tuple(
            ContourLevelEvidence(
                samples=level.samples,
                winding_number=level.winding_number,
                max_phase_increment=(
                    float(level.max_phase_increment)
                    if math.isfinite(level.max_phase_increment)
                    else None
                ),
                min_log_abs_determinant=(
                    float(level.min_log_abs_determinant)
                    if math.isfinite(level.min_log_abs_determinant)
                    else None
                ),
                finite=level.finite,
            )
            for level in evidence.levels
        )
        if not evidence.converged:
            status = "unresolved_contour_sampling"
        elif evidence.winding_number == 1:
            status = "one_isolated_zero"
        else:
            status = f"winding_count_{evidence.winding_number}"
        rows.append(
            ProbeEvidence(
                discretization=_metadata_for_profile_config(profile, config),
                status=status,
                winding_number=evidence.winding_number,
                contour_levels=levels,
                message=evidence.message,
            )
        )
    return tuple(rows)


def _tier_artifact(
    result: NumericalTierSelectionResult,
    discretizations: tuple[DiscretizationMetadata, ...],
) -> TierSelectionResult:
    return TierSelectionResult(
        tier_name=result.tier_name,
        discretizations=discretizations,
        selected_raw_indices=tuple(
            sorted(int(value) for value in result.selected_original_indices)
        ),
        status_counts=result.status_counts,
        elapsed_seconds=result.elapsed_seconds,
    )


def _numerical_failure_counts(
    roots: tuple[RootSelectionEvidence, ...],
) -> Mapping[str, int]:
    statuses = [value for item in roots for _, value in item.stage_statuses]
    messages = [probe.message.lower() for item in roots for probe in item.probes]
    return {
        "unresolved": sum(
            item.final_state is SelectorFinalState.UNRESOLVED for item in roots
        ),
        "contour": sum(
            "contour" in value and value.startswith("unresolved")
            for value in statuses
        ),
        "center": sum(
            "center" in value and value.startswith("unresolved")
            for value in statuses
        ),
        "nonfinite": sum("nonfinite" in value for value in statuses),
        "overflow": sum("overflow" in value for value in messages),
        "svd": sum("svd" in value for value in messages),
    }


def _metadata_for_profile_config(
    profile: SelectorProfile,
    config: LegacyDiscretization,
) -> DiscretizationMetadata:
    candidates = [profile.source.discretization, *profile.near.discretizations]
    if profile.fallback is not None:
        candidates.append(profile.fallback.discretization)
    if profile.core is not None:
        candidates.extend(profile.core.discretizations)
    for candidate in candidates:
        if _legacy_config(candidate) == config:
            return candidate
    raise KeyError(f"legacy discretization {config} is absent from the frozen profile")


def _legacy_config(
    metadata: DiscretizationMetadata,
) -> LegacyDiscretization:
    if metadata.family != LEGACY_FAMILY:
        raise ValueError("legacy profile contains a non-legacy discretization")
    if metadata.pole_order is None or metadata.harmonic_truncation is None:
        raise ValueError("legacy discretization requires pole_order and harmonic_truncation")
    if not metadata.variant:
        raise ValueError("legacy discretization requires a J-pole variant")
    return (
        int(metadata.pole_order),
        int(metadata.harmonic_truncation),
        str(metadata.variant),
    )


def _adapter_metadata(
    record: BOCaseRecord,
    operator: LegacyRationalOperator,
) -> tuple[str, str]:
    if isinstance(operator, LegacyESRationalOperator):
        return "legacy_es_scalar_schur", "electrostatic scalar finite-BO operator"
    if isinstance(operator, LegacyEMRationalOperator) and operator.is_darwin:
        return (
            "legacy_darwin_generalized_3x3_schur",
            "generalized Darwin pencil with explicit electric mass block",
        )
    if record.case_id == "bo2018_darwin" and record.model.iem == 1:
        return (
            "legacy_em_3x3_schur",
            "canonical iem=1 electromagnetic override; not the raw iem=2 Darwin pencil",
        )
    return "legacy_em_3x3_schur", "electromagnetic 3x3 finite-BO operator"


def _load_one_source_spectrum(path: Path, *, scan_index: int) -> np.ndarray:
    if path.suffix.lower() != ".npz":
        raise ValueError("a supplied source spectrum must be a single-scan NPZ artifact")
    with np.load(path, allow_pickle=False) as payload:
        if "normalized_roots" in payload.files:
            roots = np.asarray(payload["normalized_roots"], dtype=np.complex128)
        elif "roots_normalized" in payload.files:
            roots = np.asarray(payload["roots_normalized"], dtype=np.complex128)
        elif "roots" in payload.files and "normalization_frequency" in payload.files:
            roots = np.asarray(payload["roots"], dtype=np.complex128) / float(
                np.asarray(payload["normalization_frequency"]).reshape(())
            )
        else:
            raise ValueError(
                "source NPZ must contain normalized_roots, roots_normalized, or "
                "roots plus normalization_frequency"
            )
        roots = np.atleast_2d(roots)
        if roots.shape[0] != 1:
            raise ValueError(
                "source NPZ exposes more than one spectrum; selection admits exactly one"
            )
        if "scan_indices" in payload.files:
            stored = np.asarray(payload["scan_indices"], dtype=np.int64).reshape(-1)
            if stored.size != 1 or int(stored[0]) != scan_index:
                raise ValueError("source NPZ scan index differs from the requested scan")
    return np.asarray(roots[0], dtype=np.complex128)


def _readonly(values: np.ndarray, *, dtype=np.complex128) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _roots_near_values(
    roots: np.ndarray,
    values: list[complex],
    *,
    tolerance: float,
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Return every source root geometrically near an uncertified pole value.

    This is an abstention set, not a deletion set: no multiplicity is inferred
    from these matches and every index remains in the source-root accounting.
    """

    if not values:
        return np.empty(0, dtype=np.int64)
    source = np.asarray(roots, dtype=np.complex128).reshape(-1)
    mask = np.zeros(source.size, dtype=bool)
    for value in values:
        distances = np.abs(source - value) / (
            1.0 + np.maximum(np.abs(source), abs(value))
        )
        mask |= distances <= tolerance
    if excluded is not None:
        mask[np.asarray(excluded, dtype=np.int64)] = False
    return np.flatnonzero(mask).astype(np.int64)


def _spectrum_sha256(roots: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(roots, dtype="<c16").reshape(-1))
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _spectrum_sha256_allow_nonfinite(roots: np.ndarray) -> str:
    return _spectrum_sha256(roots)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "LegacyPoleRemoval",
    "LegacySelectionAbstained",
    "LegacySelectionRunResult",
    "select_legacy_case",
]

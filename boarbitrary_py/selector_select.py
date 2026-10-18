from __future__ import annotations

"""Label-blind BO root selection from exactly one source spectrum.

This module intentionally imports no evaluation, dataset, paper-curve, or
branch-table code.  Alternative full spectra are not accepted by its public
API.  The only large root array admitted is the single source spectrum.
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .case_registry import BOCaseRecord, get_case_record
from .certified_poles import (
    CertifiedPoleRemoval,
    remove_hh_structural_poles,
    uncertified_pole_proximity_mask,
)
from .hh_adaptive_contour_selector import (
    RootCountDecision,
    TierSelectionResult as HHTierSelectionResult,
    extend_tier_selection,
    fast_adjacent_prefilter,
    select_roots_by_local_counts,
)
from .hh_local_selector import HHRationalOperator, build_rational_operator
from .hh_minimal import build_hh_minimal_realization
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


HH_FAMILY = "2025_bo_arbitrary_hh"


@dataclass(frozen=True)
class SelectionRunResult:
    manifest: PredictionManifest
    paths: FrozenPredictionPaths
    source_spectrum_origin: str
    all_root_audit_disagreement_count: int | None


class SelectionAbstained(RuntimeError):
    """Raised after a whole-scan diagnostic has been durably committed."""

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


def select_hh_case(
    case_id: str,
    *,
    scan_index: int,
    output_dir: str | Path,
    workspace: str | Path | None = None,
    source_spectrum_path: str | Path | None = None,
    profile: SelectorProfile | None = None,
    stability_tolerance: float = 1.0e-3,
    quadrature_order: int = 80,
    include_core: bool = True,
    audit_all_roots: bool = False,
    artifact_stem: str | None = None,
) -> SelectionRunResult:
    """Run the HH selector and freeze its evidence before evaluation.

    If ``source_spectrum_path`` is omitted, this function performs one and
    only one registered full eigensolve at ``scan_index``.  A supplied NPZ
    must contain exactly one normalized spectrum row.
    """

    record = get_case_record(case_id)
    _require_hh_record(record, scan_index)
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
    _validate_hh_profile(selected_profile, record, scan_index)
    source_config = _hh_config(selected_profile.source.discretization)
    if selected_profile.fallback is None:
        raise ValueError("the HH fast selector requires one declared fallback probe")

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
    if not np.all(np.isfinite(source_roots.real) & np.isfinite(source_roots.imag)):
        raise ValueError("the immutable root-evidence schema requires a finite source spectrum")

    primary_configs = tuple(
        _hh_config(item) for item in selected_profile.near.discretizations
    )
    fallback_config = _hh_config(selected_profile.fallback.discretization)
    core_configs = (
        ()
        if not include_core or selected_profile.core is None
        else tuple(_hh_config(item) for item in selected_profile.core.discretizations)
    )
    all_configs = tuple(
        dict.fromkeys((source_config, *primary_configs, fallback_config, *core_configs))
    )

    operator_started = perf_counter()
    operators: dict[tuple[int, int], HHRationalOperator] = {
        config: build_rational_operator(
            case_id,
            workspace=workspace_root,
            scan_index=scan_index,
            j_poles=config[0],
            n_harmonics=config[1],
            quadrature_order=quadrature_order,
        )
        for config in all_configs
    }
    operator_build_seconds = perf_counter() - operator_started
    source_operator = operators[source_config]
    try:
        certified = remove_hh_structural_poles(source_roots, source_operator)
    except ValueError as exc:
        rejected_certificate = build_hh_minimal_realization(
            source_operator.coefficients
        ).certificate
        diagnostic = {
            "schema_version": 1,
            "status": "abstained",
            "reason": "structural_pole_certificate_rejected",
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "case_id": case_id,
            "scan_index_0based": scan_index,
            "source_root_count": int(source_roots.size),
            "source_spectrum_sha256": _spectrum_sha256(source_roots),
            "source_spectrum_origin": source_origin,
            "profile_id": selected_profile.profile_id,
            "error": str(exc),
            "certificate": {
                "original_size": rejected_certificate.original_size,
                "minimal_size": rejected_certificate.minimal_size,
                "known_pole_count": rejected_certificate.known_pole_count,
                "reconstructed_root_count": rejected_certificate.reconstructed_root_count,
                "count_closure": rejected_certificate.count_closure,
                "ambiguous_rank_count": rejected_certificate.ambiguous_rank_count,
                "mcmillan_minimal": rejected_certificate.mcmillan_minimal,
                "rank_tolerance": rejected_certificate.rank_tolerance,
            },
            "raw_csnj_used_as_deletion_list": False,
            "prediction_manifest_written": False,
            "scientific_interpretation": (
                "Structural-pole multiplicity was not certified, so the selector "
                "abstained for the entire scan point instead of deleting raw csnj poles."
            ),
        }
        diagnostic_stem = artifact_stem or (
            f"{case_id}__scan-{scan_index:04d}__selection-abstention"
        )
        paths = freeze_selection_diagnostic(
            diagnostic,
            output_dir,
            stem=diagnostic_stem,
        )
        raise SelectionAbstained(
            str(exc), diagnostic=diagnostic, paths=paths
        ) from exc
    # The profile artifact records the gate that actually certified this
    # spectrum.  The matcher deliberately tries increasingly permissive
    # numerical guards, so retaining the dataclass default here would falsely
    # imply that a tighter value passed.
    selected_profile = replace(
        selected_profile,
        tolerance=replace(
            selected_profile.tolerance,
            structural_pole_match_tolerance=certified.provenance.match_tolerance,
        ),
    )
    retained_roots = certified.removal.retained_roots
    retained_indices = certified.removal.retained_indices
    uncertified_pole_mask = uncertified_pole_proximity_mask(
        source_roots, certified
    )
    tolerance = selected_profile.tolerance
    contour = selected_profile.contour

    near_operator_configs = tuple(
        dict.fromkeys((source_config, fallback_config, *primary_configs))
    )
    near_operators = {config: operators[config] for config in near_operator_configs}
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
        certified,
        selected_profile,
        prefilter_status=prefilter.status_by_position,
        source_centers=prefilter.source_centers,
        near_result=near_result,
        core_result=core_result,
        audit_result=audit_result,
        uncertified_pole_mask=uncertified_pole_mask,
    )
    accounting = SpectrumAccounting.from_evidence(source_roots.size, roots)
    tier_results = [_tier_artifact(near_result, selected_profile.near.discretizations)]
    if core_result is not None and selected_profile.core is not None:
        tier_results.append(_tier_artifact(core_result, selected_profile.core.discretizations))
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
    manifest = PredictionManifest(
        selector_id="bo-fast-hh-local-count-v1",
        generated_at_utc=generated_at,
        source=SourceSpectrumProvenance(
            case_id=case_id,
            scan_index=scan_index,
            root_count=source_roots.size,
            spectrum_sha256=source_hash,
            solver_id=(
                "boarbitrary_py.unified_runner:full"
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
        family_metadata={
            "adapter": "hh_finite_bo_3x3_schur",
            "pole_certificate_kind": certified.provenance.certificate_kind,
            "pole_certificate_sha256": certified.provenance.structural_poles_sha256,
            "pole_match_tolerance": certified.provenance.match_tolerance,
            "pole_match_tolerance_semantics": "adaptive_selected_certification_gate",
            "pole_certificate_provenance_json": json.dumps(
                certified.provenance.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "uncertified_pole_proximity_root_count": int(
                np.count_nonzero(uncertified_pole_mask)
            ),
            "raw_csnj_used_as_deletion_list": False,
            "all_root_audit_requested": audit_all_roots,
            "all_root_audit_disagreement_count": audit_disagreement,
        },
        selector_input_manifest_sha256=None,
    )
    serialization_started = perf_counter()
    manifest.canonical_json()
    serialization_seconds = perf_counter() - serialization_started
    final_timing = replace(
        provisional_timing,
        prediction_serialization_seconds=serialization_seconds,
        selector_total_seconds=selector_elapsed_without_serialization + serialization_seconds,
        end_to_end_seconds=(
            source_wall_seconds + selector_elapsed_without_serialization + serialization_seconds
        ),
    )
    manifest = replace(manifest, timings=final_timing)
    paths = freeze_prediction_artifact(manifest, output_dir, stem=artifact_stem)
    return SelectionRunResult(
        manifest=manifest,
        paths=paths,
        source_spectrum_origin=source_origin,
        all_root_audit_disagreement_count=audit_disagreement,
    )


def selection_cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bo-cases select",
        description=(
            "Select BO-discretization-stable candidates from exactly one HH or "
            "legacy PDRK source spectrum."
        ),
    )
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument("--scan-index", type=int, required=True)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--source-spectrum-npz", type=Path, default=None)
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument("--stability-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--quadrature-order", type=int, default=80)
    parser.add_argument("--without-core", action="store_true")
    parser.add_argument("--audit-all-roots", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs") / "selector_predictions"
    )
    parser.add_argument("--artifact-stem", default=None)
    args = parser.parse_args(argv)
    profile = None
    if args.profile_json is not None:
        profile = SelectorProfile.from_dict(
            json.loads(args.profile_json.read_text(encoding="utf-8"))
        )
    record = get_case_record(args.case_id)
    legacy_abstention_type: type[RuntimeError] = SelectionAbstained
    if record.source_family == HH_FAMILY:
        selector = select_hh_case
    else:
        # Lazy family dispatch preserves the prediction/evaluation process
        # boundary and keeps HH regression imports unchanged.
        from .selector_legacy import LegacySelectionAbstained, select_legacy_case

        selector = select_legacy_case
        legacy_abstention_type = LegacySelectionAbstained
    try:
        result = selector(
            args.case_id,
            scan_index=args.scan_index,
            output_dir=args.output_dir,
            workspace=args.workspace,
            source_spectrum_path=args.source_spectrum_npz,
            profile=profile,
            stability_tolerance=args.stability_tolerance,
            quadrature_order=args.quadrature_order,
            include_core=not args.without_core,
            audit_all_roots=args.audit_all_roots,
            artifact_stem=args.artifact_stem,
        )
    except (SelectionAbstained, legacy_abstention_type) as exc:
        print(
            json.dumps(
                {
                    **exc.diagnostic,
                    "diagnostic_json": str(exc.paths.json_path),
                    "diagnostic_sha256": str(exc.paths.checksum_path),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "case_id": result.manifest.source.case_id,
                "scan_index_0based": result.manifest.source.scan_index,
                "source_spectrum_origin": result.source_spectrum_origin,
                "prediction_json": str(result.paths.json_path),
                "prediction_csv": str(result.paths.csv_path),
                "prediction_sha256": str(result.paths.checksum_path),
                "state_counts": result.manifest.accounting.to_dict()["state_counts"],
                "selected_index_hashes": result.manifest.selected_index_hashes,
                "all_root_audit_disagreement_count": (
                    result.all_root_audit_disagreement_count
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _require_hh_record(record: BOCaseRecord, scan_index: int) -> None:
    if record.source_family != HH_FAMILY:
        raise ValueError(
            f"{record.case_id} is {record.source_family}; its legacy selector adapter "
            "must be used instead of the HH equations"
        )
    if not 0 <= scan_index < record.scan.points:
        raise ValueError(
            f"scan_index {scan_index} is outside [0, {record.scan.points - 1}]"
        )


def _validate_hh_profile(
    profile: SelectorProfile,
    record: BOCaseRecord,
    scan_index: int,
) -> None:
    if profile.family != HH_FAMILY or profile.family != record.source_family:
        raise ValueError("selector profile family does not match the registered HH case")
    if profile.scan_scope.case_id != record.case_id or not profile.scan_scope.contains(scan_index):
        raise ValueError("selector profile scan scope does not admit the requested source point")
    source = _hh_config(profile.source.discretization)
    expected = (record.model.j_poles, record.model.n_harmonics)
    if source != expected:
        raise ValueError(
            f"selector source discretization {source} differs from registered {expected}"
        )


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


def _build_root_evidence(
    source_roots: np.ndarray,
    certified: CertifiedPoleRemoval,
    profile: SelectorProfile,
    *,
    prefilter_status: tuple[str, ...],
    source_centers: tuple[Any, ...],
    near_result: HHTierSelectionResult,
    core_result: HHTierSelectionResult | None,
    audit_result: HHTierSelectionResult | None,
    uncertified_pole_mask: np.ndarray | None = None,
) -> tuple[RootSelectionEvidence, ...]:
    pole_evidence = _certified_pole_evidence(source_roots, certified)
    retained_position = {
        int(raw_index): position
        for position, raw_index in enumerate(certified.removal.retained_indices)
    }
    near_selected = set(int(value) for value in near_result.selected_original_indices)
    core_selected = (
        set() if core_result is None else set(int(value) for value in core_result.selected_original_indices)
    )
    audit_selected = (
        None
        if audit_result is None
        else set(int(value) for value in audit_result.selected_original_indices)
    )
    near_by_position = {item.retained_position: item for item in near_result.decisions}
    core_by_position = (
        {} if core_result is None else {item.retained_position: item for item in core_result.decisions}
    )
    audit_by_position = (
        {} if audit_result is None else {item.retained_position: item for item in audit_result.decisions}
    )
    epsilon = profile.tolerance.normalized_movement_epsilon
    forced_unresolved = (
        np.zeros(source_roots.size, dtype=bool)
        if uncertified_pole_mask is None
        else np.asarray(uncertified_pole_mask, dtype=bool).reshape(-1)
    )
    if forced_unresolved.size != source_roots.size:
        raise ValueError("uncertified-pole proximity mask must match source roots")
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
                    stage_statuses={"structural_pole": "matched_mcmillan_certificate"},
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
        if forced_unresolved[raw_index]:
            statuses["uncertified_pole_multiplicity"] = "proximity_guard"
        state = _final_state(
            raw_index,
            statuses,
            near_selected=near_selected,
            core_selected=core_selected,
            audit_selected=audit_selected,
            forced_unresolved=bool(forced_unresolved[raw_index]),
        )
        evidence_decision = _best_decision(
            state,
            near_decision=near_decision,
            core_decision=core_decision,
            audit_decision=audit_decision,
        )
        probes = _probe_evidence(evidence_decision, profile)
        failed_probe = None
        if evidence_decision.first_failed_discretization is not None:
            candidate = _metadata_for_profile_config(
                profile, evidence_decision.first_failed_discretization
            )
            if candidate.key in {item.discretization.key for item in probes}:
                failed_probe = candidate
        center_info = source_centers[position]
        center = complex(center_info.center) if center_info.finite else None
        radius = (
            float(evidence_decision.radius)
            if math.isfinite(evidence_decision.radius) and evidence_decision.radius > 0.0
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


def _certified_pole_evidence(
    source_roots: np.ndarray,
    certified: CertifiedPoleRemoval,
) -> dict[int, StructuralPoleEvidence]:
    poles = np.asarray(certified.structural_poles_normalized, dtype=np.complex128)
    costs = np.abs(poles[:, None] - source_roots[None, :]) / (
        1.0 + np.maximum(np.abs(poles)[:, None], np.abs(source_roots)[None, :])
    )
    pole_rows, root_columns = linear_sum_assignment(costs)
    assignment = {int(root): int(pole) for pole, root in zip(pole_rows, root_columns)}
    removed = set(int(value) for value in certified.removal.removed_indices)
    if set(assignment) != removed:
        raise AssertionError("certified structural-pole assignment differs from removal indices")

    groups: list[complex] = []
    occurrence_group: list[tuple[int, int]] = []
    group_members: list[list[int]] = []
    for occurrence, value in enumerate(poles):
        group_index = next(
            (
                index
                for index, existing in enumerate(groups)
                if abs(value - existing) <= 1.0e-13 * (1.0 + abs(existing))
            ),
            None,
        )
        if group_index is None:
            group_index = len(groups)
            groups.append(complex(value))
            group_members.append([])
        slot = len(group_members[group_index])
        group_members[group_index].append(occurrence)
        occurrence_group.append((group_index, slot))

    result: dict[int, StructuralPoleEvidence] = {}
    certificate = certified.provenance
    for raw_index, occurrence in assignment.items():
        group_index, slot = occurrence_group[occurrence]
        declared = DeclaredPole(
            value_normalized=groups[group_index],
            multiplicity=len(group_members[group_index]),
            provenance=certificate.certificate_kind,
            certificate_id=(
                f"{certificate.structural_poles_sha256}:group-{group_index:04d}"
            ),
        )
        result[raw_index] = StructuralPoleEvidence(
            matched=True,
            declared_pole=declared,
            matching_distance=float(costs[occurrence, raw_index]),
            multiplicity_slot=slot,
            message=(
                "matched by global assignment to a McMillan-certified pole "
                f"multiplicity; guard={certificate.match_tolerance:.3e}"
            ),
        )
    return result


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
    result: HHTierSelectionResult,
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
        "contour": sum("contour" in value and value.startswith("unresolved") for value in statuses),
        "center": sum("center" in value and value.startswith("unresolved") for value in statuses),
        "nonfinite": sum("nonfinite" in value for value in statuses),
        "overflow": sum("overflow" in value for value in messages),
        "svd": sum("svd" in value for value in messages),
    }


def _metadata_for_config(config: tuple[int, int]) -> DiscretizationMetadata:
    return DiscretizationMetadata(
        family=HH_FAMILY,
        pole_order=int(config[0]),
        harmonic_truncation=int(config[1]),
        variant="finite_bo",
    )


def _metadata_for_profile_config(
    profile: SelectorProfile,
    config: tuple[int, int],
) -> DiscretizationMetadata:
    candidates = [profile.source.discretization, *profile.near.discretizations]
    if profile.fallback is not None:
        candidates.append(profile.fallback.discretization)
    if profile.core is not None:
        candidates.extend(profile.core.discretizations)
    for candidate in candidates:
        if _hh_config(candidate) == config:
            return candidate
    return _metadata_for_config(config)


def _hh_config(metadata: DiscretizationMetadata) -> tuple[int, int]:
    if metadata.family != HH_FAMILY:
        raise ValueError("HH profile contains a non-HH discretization")
    if metadata.pole_order is None or metadata.harmonic_truncation is None:
        raise ValueError("HH discretization requires pole_order and harmonic_truncation")
    return int(metadata.pole_order), int(metadata.harmonic_truncation)


def _spectrum_sha256(roots: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(roots, dtype="<c16").reshape(-1))
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SelectionAbstained",
    "SelectionRunResult",
    "select_hh_case",
    "selection_cli_main",
]

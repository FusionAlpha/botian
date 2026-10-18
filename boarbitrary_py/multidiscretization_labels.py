from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


class PilotLabel(StrEnum):
    STRICT_POSITIVE = "pilot_strict_positive"
    PROBABLE_POSITIVE = "pilot_probable_positive"
    UNRESOLVED = "unresolved"


class CoverageClass(StrEnum):
    STRICT = "strict_coverage"
    PROBABLE = "probable_coverage"
    INSUFFICIENT = "insufficient_coverage"


@dataclass(frozen=True)
class DiscretizationSpec:
    discretization_id: str
    j_poles: int
    n_harmonics: int

    def __post_init__(self) -> None:
        if not self.discretization_id:
            raise ValueError("discretization_id must be non-empty")
        if self.j_poles <= 0:
            raise ValueError("j_poles must be positive")
        if self.n_harmonics < 0:
            raise ValueError("n_harmonics must be nonnegative")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "discretization_id": self.discretization_id,
            "J": int(self.j_poles),
            "N": int(self.n_harmonics),
        }


@dataclass(frozen=True)
class PoleRemovalResult:
    retained_roots: np.ndarray
    retained_indices: np.ndarray
    removed_roots: np.ndarray
    removed_indices: np.ndarray
    declared_pole_count: int
    nonfinite_root_count: int
    max_normalized_match_distance: float

    def to_manifest(self) -> dict[str, Any]:
        return {
            "declared_pole_count": int(self.declared_pole_count),
            "removed_count": int(self.removed_roots.size),
            "retained_count": int(self.retained_roots.size),
            "nonfinite_root_count": int(self.nonfinite_root_count),
            "max_normalized_match_distance": float(self.max_normalized_match_distance),
            "removed_indices_0based": [int(value) for value in self.removed_indices],
        }


@dataclass(frozen=True)
class RootObservation:
    discretization_id: str
    root_index: int
    root: complex
    normalized_match_distance: float

    def to_manifest(self) -> dict[str, Any]:
        return {
            "discretization_id": self.discretization_id,
            "root_index_0based": int(self.root_index),
            "root": _complex_manifest(self.root),
            "normalized_match_distance": float(self.normalized_match_distance),
        }


@dataclass(frozen=True)
class AnchoredRootCluster:
    candidate_id: str
    reference_discretization_id: str
    reference_root_index: int
    reference_root: complex
    centroid: complex
    observations: tuple[RootObservation, ...]
    represented_discretizations: tuple[str, ...]
    represented_j_levels: tuple[int, ...]
    represented_n_levels: tuple[int, ...]
    max_normalized_match_distance: float
    normalized_spread: float
    isolation_radius: float
    ambiguous: bool

    def to_manifest(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reference_discretization_id": self.reference_discretization_id,
            "reference_root_index_0based": int(self.reference_root_index),
            "reference_root": _complex_manifest(self.reference_root),
            "centroid": _complex_manifest(self.centroid),
            "observations": [item.to_manifest() for item in self.observations],
            "represented_discretizations": list(self.represented_discretizations),
            "represented_j_levels": [int(value) for value in self.represented_j_levels],
            "represented_n_levels": [int(value) for value in self.represented_n_levels],
            "configuration_coverage": int(len(self.represented_discretizations)),
            "max_normalized_match_distance": float(self.max_normalized_match_distance),
            "normalized_spread": float(self.normalized_spread),
            "isolation_radius": None if not np.isfinite(self.isolation_radius) else float(self.isolation_radius),
            "ambiguous": bool(self.ambiguous),
        }


@dataclass(frozen=True)
class CoverageRequirements:
    strict_min_configurations: int = 6
    strict_min_j_levels: int = 3
    strict_min_n_levels: int = 3
    probable_min_configurations: int = 4
    probable_min_j_levels: int = 3
    probable_min_n_levels: int = 2


@dataclass(frozen=True)
class ExactValidationEvidence:
    attempted_seed_count: int
    accepted_seed_count: int
    residual_qualified_seed_count: int
    accepted_harmonic_levels: tuple[int, ...]
    corrected_roots: tuple[complex, ...]
    max_normalized_disagreement: float
    independent_seed_agreement: bool
    high_n_agreement: bool

    @classmethod
    def empty(cls) -> ExactValidationEvidence:
        return cls(
            attempted_seed_count=0,
            accepted_seed_count=0,
            residual_qualified_seed_count=0,
            accepted_harmonic_levels=(),
            corrected_roots=(),
            max_normalized_disagreement=float("inf"),
            independent_seed_agreement=False,
            high_n_agreement=False,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "attempted_seed_count": int(self.attempted_seed_count),
            "accepted_seed_count": int(self.accepted_seed_count),
            "residual_qualified_seed_count": int(self.residual_qualified_seed_count),
            "accepted_harmonic_levels": [int(value) for value in self.accepted_harmonic_levels],
            "corrected_roots": [_complex_manifest(value) for value in self.corrected_roots],
            "max_normalized_disagreement": (
                None if not np.isfinite(self.max_normalized_disagreement) else float(self.max_normalized_disagreement)
            ),
            "independent_seed_agreement": bool(self.independent_seed_agreement),
            "high_n_agreement": bool(self.high_n_agreement),
        }


@dataclass(frozen=True)
class PilotLabelDecision:
    label: PilotLabel
    coverage_class: CoverageClass
    reasons: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "label": self.label.value,
            "coverage_class": self.coverage_class.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RefinementPathEvidence:
    reference_discretization_id: str
    reference_root_index: int
    required_discretizations: tuple[str, ...]
    observations: tuple[RootObservation, ...]
    complete: bool
    required_match_radius: float
    uniqueness_radius: float
    certificate_interval_nonempty: bool

    def to_manifest(self) -> dict[str, Any]:
        return {
            "reference_discretization_id": self.reference_discretization_id,
            "reference_root_index_0based": int(self.reference_root_index),
            "required_discretizations": list(self.required_discretizations),
            "observations": [item.to_manifest() for item in self.observations],
            "complete": bool(self.complete),
            "required_match_radius": float(self.required_match_radius),
            "uniqueness_radius": None if not np.isfinite(self.uniqueness_radius) else float(self.uniqueness_radius),
            "certificate_interval_nonempty": bool(self.certificate_interval_nonempty),
            "certificate_tolerance_interval": (
                [float(self.required_match_radius), float(self.uniqueness_radius)]
                if self.certificate_interval_nonempty
                else None
            ),
        }


@dataclass(frozen=True)
class ModeLineagePoint:
    layer_index: int
    coordinate: float
    root_index: int
    root: complex
    predicted_root: complex
    normalized_prediction_error: float

    def to_manifest(self) -> dict[str, Any]:
        return {
            "layer_index": int(self.layer_index),
            "coordinate": float(self.coordinate),
            "root_index_0based": int(self.root_index),
            "root": _complex_manifest(self.root),
            "predicted_root": _complex_manifest(self.predicted_root),
            "normalized_prediction_error": float(self.normalized_prediction_error),
        }


@dataclass(frozen=True)
class ModeLineage:
    mode_spec: str
    points: tuple[ModeLineagePoint, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mode_spec": self.mode_spec,
            "points": [point.to_manifest() for point in self.points],
            "max_normalized_prediction_error": max(
                (point.normalized_prediction_error for point in self.points),
                default=0.0,
            ),
        }


def remove_declared_poles(
    roots: Sequence[complex] | np.ndarray,
    known_poles: Sequence[complex] | np.ndarray,
    *,
    known_count: int | None = None,
    match_tolerance: float = 1.0e-12,
) -> PoleRemovalResult:
    if match_tolerance <= 0.0:
        raise ValueError("match_tolerance must be positive")
    root_values = np.asarray(roots, dtype=np.complex128).reshape(-1)
    finite_root_mask = _finite_complex_mask(root_values)
    finite_indices = np.flatnonzero(finite_root_mask)
    finite_roots = root_values[finite_root_mask]

    pole_values = np.asarray(known_poles, dtype=np.complex128).reshape(-1)
    if known_count is not None:
        if known_count < 0 or known_count > pole_values.size:
            raise ValueError("known_count is outside the known_poles array")
        pole_values = pole_values[:known_count]
    pole_values = pole_values[_finite_complex_mask(pole_values)]
    if pole_values.size == 0:
        return PoleRemovalResult(
            retained_roots=finite_roots,
            retained_indices=finite_indices,
            removed_roots=np.empty(0, dtype=np.complex128),
            removed_indices=np.empty(0, dtype=np.int64),
            declared_pole_count=0,
            nonfinite_root_count=int(root_values.size - finite_roots.size),
            max_normalized_match_distance=0.0,
        )
    if pole_values.size > finite_roots.size:
        raise ValueError("declared pole multiplicity exceeds the finite spectrum size")

    costs = _normalized_distance_matrix(pole_values, finite_roots)
    pole_rows, root_columns = linear_sum_assignment(costs)
    matched_costs = costs[pole_rows, root_columns]
    max_distance = float(np.max(matched_costs))
    if max_distance > match_tolerance:
        raise ValueError(
            f"declared poles are inconsistent with the spectrum: max normalized match "
            f"distance {max_distance:.3e} exceeds {match_tolerance:.3e}"
        )

    removed_mask = np.zeros(finite_roots.size, dtype=bool)
    removed_mask[root_columns] = True
    return PoleRemovalResult(
        retained_roots=finite_roots[~removed_mask],
        retained_indices=finite_indices[~removed_mask],
        removed_roots=finite_roots[root_columns],
        removed_indices=finite_indices[root_columns],
        declared_pole_count=int(pole_values.size),
        nonfinite_root_count=int(root_values.size - finite_roots.size),
        max_normalized_match_distance=max_distance,
    )


def build_reference_anchored_clusters(
    spectra: Mapping[str, Sequence[complex] | np.ndarray],
    specs: Mapping[str, DiscretizationSpec],
    *,
    reference_id: str,
    match_tolerance: float,
) -> tuple[AnchoredRootCluster, ...]:
    if match_tolerance <= 0.0:
        raise ValueError("match_tolerance must be positive")
    if reference_id not in spectra or reference_id not in specs:
        raise ValueError("reference_id must exist in both spectra and specs")
    if set(spectra) != set(specs):
        raise ValueError("spectra and specs must use the same discretization ids")

    finite_spectra: dict[str, np.ndarray] = {}
    finite_indices: dict[str, np.ndarray] = {}
    for discretization_id, values in spectra.items():
        roots = np.asarray(values, dtype=np.complex128).reshape(-1)
        mask = _finite_complex_mask(roots)
        finite_spectra[discretization_id] = roots[mask]
        finite_indices[discretization_id] = np.flatnonzero(mask)

    reference_roots = finite_spectra[reference_id]
    observations: list[list[RootObservation]] = [
        [
            RootObservation(
                discretization_id=reference_id,
                root_index=int(finite_indices[reference_id][index]),
                root=complex(root),
                normalized_match_distance=0.0,
            )
        ]
        for index, root in enumerate(reference_roots)
    ]
    competitor_distances = _nearest_other_distances(reference_roots)

    for discretization_id in sorted(spectra):
        if discretization_id == reference_id:
            continue
        target_roots = finite_spectra[discretization_id]
        if reference_roots.size == 0 or target_roots.size == 0:
            continue
        costs = _normalized_distance_matrix(reference_roots, target_roots)
        reference_rows, target_columns = linear_sum_assignment(costs)
        for reference_row, target_column in zip(reference_rows, target_columns, strict=True):
            distance = float(costs[reference_row, target_column])
            row_costs = costs[reference_row]
            if row_costs.size > 1:
                second_distance = float(np.partition(row_costs, 1)[1])
                competitor_distances[reference_row] = min(
                    competitor_distances[reference_row],
                    second_distance,
                )
            if distance > match_tolerance:
                continue
            observations[reference_row].append(
                RootObservation(
                    discretization_id=discretization_id,
                    root_index=int(finite_indices[discretization_id][target_column]),
                    root=complex(target_roots[target_column]),
                    normalized_match_distance=distance,
                )
            )

    clusters: list[AnchoredRootCluster] = []
    for reference_index, members in enumerate(observations):
        member_roots = np.asarray([item.root for item in members], dtype=np.complex128)
        centroid = complex(np.mean(member_roots))
        represented = tuple(sorted(item.discretization_id for item in members))
        j_levels = tuple(sorted({specs[item].j_poles for item in represented}))
        n_levels = tuple(sorted({specs[item].n_harmonics for item in represented}))
        max_distance = max(item.normalized_match_distance for item in members)
        spread = max((_normalized_root_distance(root, centroid) for root in member_roots), default=0.0)
        isolation_radius = 0.5 * float(competitor_distances[reference_index])
        clusters.append(
            AnchoredRootCluster(
                candidate_id=f"candidate_{reference_index:04d}",
                reference_discretization_id=reference_id,
                reference_root_index=int(reference_index),
                reference_root=complex(reference_roots[reference_index]),
                centroid=centroid,
                observations=tuple(members),
                represented_discretizations=represented,
                represented_j_levels=j_levels,
                represented_n_levels=n_levels,
                max_normalized_match_distance=float(max_distance),
                normalized_spread=float(spread),
                isolation_radius=isolation_radius,
                ambiguous=bool(match_tolerance >= isolation_radius),
            )
        )
    return tuple(clusters)


def build_refinement_path_evidence(
    spectra: Mapping[str, Sequence[complex] | np.ndarray],
    *,
    reference_id: str,
    reference_root_index: int,
    required_discretizations: Sequence[str],
) -> RefinementPathEvidence:
    required = tuple(dict.fromkeys(str(value) for value in required_discretizations))
    if not required:
        raise ValueError("required_discretizations must not be empty")
    if reference_id not in spectra or reference_id not in required:
        raise ValueError("the reference discretization must be present and required")
    missing = [value for value in required if value not in spectra]
    if missing:
        raise ValueError(f"required discretizations are missing from spectra: {missing}")

    finite_spectra = {
        key: np.asarray(values, dtype=np.complex128).reshape(-1)[
            _finite_complex_mask(np.asarray(values, dtype=np.complex128).reshape(-1))
        ]
        for key, values in spectra.items()
    }
    finite_indices = {
        key: np.flatnonzero(_finite_complex_mask(np.asarray(values, dtype=np.complex128).reshape(-1)))
        for key, values in spectra.items()
    }
    reference_roots = finite_spectra[reference_id]
    if reference_root_index < 0 or reference_root_index >= reference_roots.size:
        raise ValueError("reference_root_index is outside the finite reference spectrum")
    reference_root = complex(reference_roots[reference_root_index])
    observations: list[RootObservation] = []
    competitor_distances: list[float] = []
    complete = True

    for discretization_id in required:
        roots = finite_spectra[discretization_id]
        if roots.size == 0:
            complete = False
            continue
        distances = _normalized_distance_matrix(
            np.asarray([reference_root], dtype=np.complex128),
            roots,
        )[0]
        nearest = int(np.argmin(distances))
        observations.append(
            RootObservation(
                discretization_id=discretization_id,
                root_index=int(finite_indices[discretization_id][nearest]),
                root=complex(roots[nearest]),
                normalized_match_distance=float(distances[nearest]),
            )
        )
        if roots.size > 1:
            competitor_distances.append(float(np.partition(distances, 1)[1]))

    reference_competitor = _nearest_other_distances(reference_roots)[reference_root_index]
    competitor_distances.append(float(reference_competitor))
    required_radius = max((item.normalized_match_distance for item in observations), default=float("inf"))
    uniqueness_radius = 0.5 * min(competitor_distances, default=float("inf"))
    certificate = bool(complete and len(observations) == len(required) and required_radius < uniqueness_radius)
    return RefinementPathEvidence(
        reference_discretization_id=reference_id,
        reference_root_index=int(reference_root_index),
        required_discretizations=required,
        observations=tuple(observations),
        complete=bool(complete and len(observations) == len(required)),
        required_match_radius=float(required_radius),
        uniqueness_radius=float(uniqueness_radius),
        certificate_interval_nonempty=certificate,
    )


def track_dominant_mode_lineage(
    root_layers: Sequence[Sequence[complex] | np.ndarray],
    coordinates: Sequence[float],
    *,
    minimum_root_magnitude: float = 1.0e-4,
) -> ModeLineage:
    if len(root_layers) == 0 or len(root_layers) != len(coordinates):
        raise ValueError("root_layers and coordinates must have the same nonzero length")
    if minimum_root_magnitude < 0.0:
        raise ValueError("minimum_root_magnitude must be nonnegative")

    points: list[ModeLineagePoint] = []
    for layer_index, (raw_roots, coordinate) in enumerate(zip(root_layers, coordinates, strict=True)):
        roots = np.asarray(raw_roots, dtype=np.complex128).reshape(-1)
        valid = _finite_complex_mask(roots) & (np.abs(roots) >= minimum_root_magnitude)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            raise ValueError(f"layer {layer_index} has no eligible nonzero roots")

        if layer_index == 0:
            max_imag = float(np.max(roots[indices].imag))
            imag_tolerance = 1.0e-10 * max(1.0, abs(max_imag))
            tied = indices[np.abs(roots[indices].imag - max_imag) <= imag_tolerance]
            selected = min(
                (int(value) for value in tied),
                key=lambda value: (abs(roots[value].real), roots[value].real < 0.0, value),
            )
            predicted = complex(roots[selected])
            error = 0.0
        else:
            previous = points[-1]
            predicted = previous.root
            if layer_index >= 2:
                earlier = points[-2]
                delta = previous.coordinate - earlier.coordinate
                if delta != 0.0:
                    ratio = (float(coordinate) - previous.coordinate) / delta
                    predicted = previous.root + ratio * (previous.root - earlier.root)
            distances = np.asarray(
                [_normalized_root_distance(roots[index], predicted) for index in indices],
                dtype=float,
            )
            minimum = float(np.min(distances))
            tied = indices[np.abs(distances - minimum) <= 1.0e-10 * max(1.0, minimum)]
            selected = min(
                (int(value) for value in tied),
                key=lambda value: (roots[value].real < 0.0, abs(roots[value] - predicted), value),
            )
            error = _normalized_root_distance(roots[selected], predicted)
        points.append(
            ModeLineagePoint(
                layer_index=int(layer_index),
                coordinate=float(coordinate),
                root_index=int(selected),
                root=complex(roots[selected]),
                predicted_root=complex(predicted),
                normalized_prediction_error=float(error),
            )
        )
    return ModeLineage(
        mode_spec=(
            "highest-growth finite nonzero root at the first coordinate; "
            "secant-predicted complex-frequency continuation thereafter"
        ),
        points=tuple(points),
    )


def classify_coverage(
    cluster: AnchoredRootCluster,
    *,
    requirements: CoverageRequirements | None = None,
) -> CoverageClass:
    requirements = requirements or CoverageRequirements()
    configuration_count = len(cluster.represented_discretizations)
    j_count = len(cluster.represented_j_levels)
    n_count = len(cluster.represented_n_levels)
    if (
        not cluster.ambiguous
        and configuration_count >= requirements.strict_min_configurations
        and j_count >= requirements.strict_min_j_levels
        and n_count >= requirements.strict_min_n_levels
    ):
        return CoverageClass.STRICT
    if (
        not cluster.ambiguous
        and configuration_count >= requirements.probable_min_configurations
        and j_count >= requirements.probable_min_j_levels
        and n_count >= requirements.probable_min_n_levels
    ):
        return CoverageClass.PROBABLE
    return CoverageClass.INSUFFICIENT


def assign_pilot_label(
    cluster: AnchoredRootCluster,
    exact: ExactValidationEvidence,
    *,
    requirements: CoverageRequirements | None = None,
    refinement_path: RefinementPathEvidence | None = None,
) -> PilotLabelDecision:
    coverage = classify_coverage(cluster, requirements=requirements)
    if refinement_path is not None and refinement_path.certificate_interval_nonempty:
        coverage = CoverageClass.STRICT
    accepted_levels = set(exact.accepted_harmonic_levels)
    strict_exact = bool(
        exact.accepted_seed_count >= 2
        and {3, 4}.issubset(accepted_levels)
        and exact.independent_seed_agreement
        and exact.high_n_agreement
    )
    probable_exact = bool(
        exact.accepted_seed_count >= 1
        or (
            exact.residual_qualified_seed_count >= 2
            and exact.independent_seed_agreement
            and exact.high_n_agreement
        )
    )

    if coverage == CoverageClass.STRICT and strict_exact:
        return PilotLabelDecision(
            label=PilotLabel.STRICT_POSITIVE,
            coverage_class=coverage,
            reasons=(
                (
                    "a nonempty unique refinement-path tolerance interval"
                    if refinement_path is not None and refinement_path.certificate_interval_nonempty
                    else "broad independent J/N coverage"
                ),
                "isolated one-to-one cross-discretization matches",
                "accepted exact-Z corrections at N=3 and N=4",
                "independent seeds and high-N corrections agree",
            ),
        )
    if coverage in {CoverageClass.STRICT, CoverageClass.PROBABLE} and probable_exact:
        return PilotLabelDecision(
            label=PilotLabel.PROBABLE_POSITIVE,
            coverage_class=coverage,
            reasons=(
                "multi-discretization coverage is sufficient for a pilot candidate",
                "at least one exact-Z correction is accepted",
                "strict cross-refinement evidence is incomplete",
            ),
        )

    reasons: list[str] = []
    if coverage == CoverageClass.INSUFFICIENT:
        reasons.append("independent J/N coverage or isolation is insufficient")
    if not probable_exact:
        reasons.append("no accepted exact-Z correction is available")
    if probable_exact and not strict_exact:
        reasons.append("independent N=3/N=4 correction agreement is incomplete")
    return PilotLabelDecision(
        label=PilotLabel.UNRESOLVED,
        coverage_class=coverage,
        reasons=tuple(reasons),
    )


def summarize_threshold_sensitivity(
    spectra: Mapping[str, Sequence[complex] | np.ndarray],
    specs: Mapping[str, DiscretizationSpec],
    *,
    reference_id: str,
    tolerances: Sequence[float],
    exact_by_reference_index: Mapping[int, ExactValidationEvidence] | None = None,
    requirements: CoverageRequirements | None = None,
) -> tuple[dict[str, Any], ...]:
    if not tolerances:
        raise ValueError("tolerances must contain at least one value")
    exact_by_reference_index = exact_by_reference_index or {}
    rows: list[dict[str, Any]] = []
    for tolerance in tolerances:
        clusters = build_reference_anchored_clusters(
            spectra,
            specs,
            reference_id=reference_id,
            match_tolerance=float(tolerance),
        )
        decisions = [
            assign_pilot_label(
                cluster,
                exact_by_reference_index.get(cluster.reference_root_index, ExactValidationEvidence.empty()),
                requirements=requirements,
            )
            for cluster in clusters
        ]
        rows.append(
            {
                "match_tolerance": float(tolerance),
                "coverage_counts": {
                    value.value: sum(decision.coverage_class == value for decision in decisions)
                    for value in CoverageClass
                },
                "label_counts": {
                    value.value: sum(decision.label == value for decision in decisions)
                    for value in PilotLabel
                },
                "strict_candidate_ids": [
                    cluster.candidate_id
                    for cluster, decision in zip(clusters, decisions, strict=True)
                    if decision.label == PilotLabel.STRICT_POSITIVE
                ],
                "probable_candidate_ids": [
                    cluster.candidate_id
                    for cluster, decision in zip(clusters, decisions, strict=True)
                    if decision.label == PilotLabel.PROBABLE_POSITIVE
                ],
            }
        )
    return tuple(rows)


def _normalized_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.complex128).reshape(-1)
    right = np.asarray(right, dtype=np.complex128).reshape(-1)
    scale = 1.0 + np.maximum(np.abs(left)[:, None], np.abs(right)[None, :])
    return np.abs(left[:, None] - right[None, :]) / scale


def _nearest_other_distances(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128).reshape(-1)
    if values.size <= 1:
        return np.full(values.size, np.inf, dtype=float)
    distances = _normalized_distance_matrix(values, values)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def _normalized_root_distance(left: complex, right: complex) -> float:
    return float(abs(complex(left) - complex(right)) / (1.0 + max(abs(complex(left)), abs(complex(right)))))


def _finite_complex_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values.real) & np.isfinite(values.imag)


def _complex_manifest(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]

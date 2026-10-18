from __future__ import annotations

"""Adaptive local root-count filtering for BO HH spectra.

This module implements a *numerical* selector for roots of the HH BO
linearization without solving a second large eigenvalue problem.  Starting
from one finite spectrum, it evaluates only the corresponding 3x3 rational
Schur operators at nearby ``(J, N)`` discretizations and asks whether the
same small disk contains exactly one zero for every requested operator.

The selector is intentionally label-free: it never imports datasets,
reference roots, or multi-discretization spectra.  Its tolerance has a direct
meaning,

    |delta omega| / (1 + |omega|) <= stability_tolerance,

rather than being a fitted residual cutoff.

The argument-principle check is adaptively sampled and is therefore strong
numerical evidence, not an interval-arithmetic proof.  Every returned result
sets ``rigorous=False`` so that downstream code cannot silently promote it to
an exact Rouche certificate.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from .hh_local_selector import HHRationalOperator, LocalHHOperator


Array = np.ndarray
Discretization = tuple[int, int]


@dataclass(frozen=True)
class ProjectedNewtonCenter:
    initial_root: complex
    center: complex
    step: complex
    finite: bool
    normalized_sigma_min: float
    projected_derivative_abs: float
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_root": _complex_pair(self.initial_root),
            "center": _complex_pair(self.center),
            "step": _complex_pair(self.step),
            "finite": bool(self.finite),
            "normalized_sigma_min": float(self.normalized_sigma_min),
            "projected_derivative_abs": float(self.projected_derivative_abs),
            "message": self.message,
        }


@dataclass(frozen=True)
class WindingLevel:
    samples: int
    winding_number: int | None
    max_phase_increment: float
    min_log_abs_determinant: float
    finite: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": int(self.samples),
            "winding_number": self.winding_number,
            "max_phase_increment": float(self.max_phase_increment),
            "min_log_abs_determinant": float(self.min_log_abs_determinant),
            "finite": bool(self.finite),
        }


@dataclass(frozen=True)
class AdaptiveWindingEvidence:
    converged: bool
    winding_number: int | None
    samples_used: int
    max_phase_increment: float
    min_log_abs_determinant: float
    message: str
    levels: tuple[WindingLevel, ...]
    rigorous: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": bool(self.converged),
            "winding_number": self.winding_number,
            "samples_used": int(self.samples_used),
            "max_phase_increment": float(self.max_phase_increment),
            "min_log_abs_determinant": float(self.min_log_abs_determinant),
            "message": self.message,
            "levels": [level.to_dict() for level in self.levels],
            "rigorous": False,
        }


@dataclass(frozen=True)
class RootCountDecision:
    retained_position: int
    original_root_index: int
    source_root: complex
    center: complex
    radius: float
    status: str
    first_failed_discretization: Discretization | None
    center_evidence: ProjectedNewtonCenter
    contour_evidence: tuple[tuple[Discretization, AdaptiveWindingEvidence], ...]
    rigorous: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "retained_position_0based": int(self.retained_position),
            "original_root_index_0based": int(self.original_root_index),
            "source_root": _complex_pair(self.source_root),
            "center": _complex_pair(self.center),
            "radius": float(self.radius),
            "status": self.status,
            "first_failed_discretization": (
                None
                if self.first_failed_discretization is None
                else {
                    "J": int(self.first_failed_discretization[0]),
                    "N": int(self.first_failed_discretization[1]),
                }
            ),
            "center_evidence": self.center_evidence.to_dict(),
            "contour_evidence": [
                {
                    "J": int(config[0]),
                    "N": int(config[1]),
                    "evidence": evidence.to_dict(),
                }
                for config, evidence in self.contour_evidence
            ],
            "rigorous": False,
        }


@dataclass(frozen=True)
class TierSelectionResult:
    tier_name: str
    discretizations: tuple[Discretization, ...]
    stability_tolerance: float
    elapsed_seconds: float
    selected_positions: Array
    selected_original_indices: Array
    status_counts: Mapping[str, int]
    decisions: tuple[RootCountDecision, ...]
    rigorous: bool = False

    def to_dict(self, *, include_decisions: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "tier_name": self.tier_name,
            "discretizations": [
                {"J": int(config[0]), "N": int(config[1])}
                for config in self.discretizations
            ],
            "stability_tolerance": float(self.stability_tolerance),
            "elapsed_seconds": float(self.elapsed_seconds),
            "selected_count": int(self.selected_positions.size),
            "selected_retained_positions_0based": [
                int(value) for value in self.selected_positions
            ],
            "selected_original_indices_0based": [
                int(value) for value in self.selected_original_indices
            ],
            "status_counts": {key: int(value) for key, value in self.status_counts.items()},
            "rigorous": False,
        }
        if include_decisions:
            payload["decisions"] = [decision.to_dict() for decision in self.decisions]
        return payload



@dataclass(frozen=True)
class FastPrefilterResult:
    source_discretization: Discretization
    primary_probes: tuple[Discretization, ...]
    fallback_probe: Discretization
    stability_tolerance: float
    elapsed_seconds: float
    base_candidate_positions: Array
    direct_pass_positions: Array
    ambiguous_positions: Array
    selected_positions: Array
    selected_original_indices: Array
    source_centers: tuple[ProjectedNewtonCenter, ...]
    status_by_position: tuple[str, ...]
    unresolved_count: int
    rigorous: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_discretization": {
                "J": int(self.source_discretization[0]),
                "N": int(self.source_discretization[1]),
            },
            "primary_probes": [
                {"J": int(config[0]), "N": int(config[1])}
                for config in self.primary_probes
            ],
            "fallback_probe": {
                "J": int(self.fallback_probe[0]),
                "N": int(self.fallback_probe[1]),
            },
            "stability_tolerance": float(self.stability_tolerance),
            "elapsed_seconds": float(self.elapsed_seconds),
            "base_candidate_count": int(self.base_candidate_positions.size),
            "direct_pass_count": int(self.direct_pass_positions.size),
            "ambiguous_count": int(self.ambiguous_positions.size),
            "selected_count": int(self.selected_positions.size),
            "base_candidate_positions_0based": [int(v) for v in self.base_candidate_positions],
            "direct_pass_positions_0based": [int(v) for v in self.direct_pass_positions],
            "ambiguous_positions_0based": [int(v) for v in self.ambiguous_positions],
            "selected_positions_0based": [int(v) for v in self.selected_positions],
            "selected_original_indices_0based": [int(v) for v in self.selected_original_indices],
            "status_by_position": list(self.status_by_position),
            "unresolved_count": int(self.unresolved_count),
            "rigorous": False,
        }

def projected_newton_center(
    operator: LocalHHOperator,
    root_normalized: complex,
    *,
    isolation_radius: float | None = None,
    derivative_step: float = 2.0e-6,
) -> ProjectedNewtonCenter:
    """Take one SVD-projected Newton step in normalized-frequency units.

    The correction removes ordinary full-eigensolver backward error before a
    small contour is drawn.  No residual threshold is used for classification;
    the returned residual is diagnostic only.  Finite-BO HH operators expose
    an analytic derivative; protocol-compatible fixtures fall back to a
    centered finite difference.  An optional isolation radius makes a basin
    exit explicit without silently clipping the correction onto another root.
    """

    z = complex(root_normalized)
    if z == 0.0 or not _finite_complex(z):
        return ProjectedNewtonCenter(
            initial_root=z,
            center=z,
            step=0.0j,
            finite=False,
            normalized_sigma_min=float("inf"),
            projected_derivative_abs=0.0,
            message="operator is singular or non-finite at omega=0",
        )

    try:
        with np.errstate(all="ignore"):
            tensor = np.asarray(operator.tensor(z), dtype=np.complex128)
            analytic_derivative = getattr(operator, "tensor_derivative", None)
            if callable(analytic_derivative):
                derivative = np.asarray(analytic_derivative(z), dtype=np.complex128)
            else:
                h = derivative_step * (1.0 + abs(z))
                if not np.isfinite(h) or h <= 0.0:
                    raise FloatingPointError("invalid finite-difference derivative step")
                derivative = (
                    np.asarray(operator.tensor(z + h), dtype=np.complex128)
                    - np.asarray(operator.tensor(z - h), dtype=np.complex128)
                ) / (2.0 * h)

        if not _finite_matrix(tensor) or not _finite_matrix(derivative):
            raise FloatingPointError("non-finite reduced operator or derivative")

        left, singular_values, vh = np.linalg.svd(tensor)
        u = left[:, -1]
        v = vh.conj().T[:, -1]
        numerator = np.vdot(u, tensor @ v)
        denominator = np.vdot(u, derivative @ v)
        denominator_abs = float(abs(denominator))
        tensor_norm = float(singular_values[0])
        normalized_sigma = float(singular_values[-1] / (1.0 + tensor_norm))
        if not np.isfinite(denominator) or denominator_abs <= np.finfo(float).tiny:
            raise FloatingPointError("projected derivative is singular")
        step = complex(-numerator / denominator)
        center = complex(z + step)
        if not np.isfinite(step) or not np.isfinite(center):
            raise FloatingPointError("projected Newton step is non-finite")
        if isolation_radius is not None:
            if not np.isfinite(isolation_radius) or isolation_radius <= 0.0:
                raise ValueError("isolation_radius must be positive and finite")
            if abs(step) >= isolation_radius:
                return ProjectedNewtonCenter(
                    initial_root=z,
                    center=z,
                    step=step,
                    finite=False,
                    normalized_sigma_min=normalized_sigma,
                    projected_derivative_abs=denominator_abs,
                    message="projected Newton center leaves source isolation disk",
                )
        return ProjectedNewtonCenter(
            initial_root=z,
            center=center,
            step=step,
            finite=True,
            normalized_sigma_min=normalized_sigma,
            projected_derivative_abs=denominator_abs,
            message="one projected Newton step completed",
        )
    except (FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError, ValueError) as exc:
        return ProjectedNewtonCenter(
            initial_root=z,
            center=z,
            step=0.0j,
            finite=False,
            normalized_sigma_min=float("inf"),
            projected_derivative_abs=0.0,
            message=str(exc),
        )


def adaptive_winding_number(
    operator: LocalHHOperator,
    center: complex,
    radius: float,
    *,
    initial_samples: int = 16,
    maximum_samples: int = 128,
    maximum_phase_increment: float = np.pi / 2.0,
) -> AdaptiveWindingEvidence:
    """Numerically count zeros-minus-poles inside one circular contour.

    Sampling doubles until two consecutive winding counts agree and every
    adjacent determinant-phase increment is smaller than the requested phase
    resolution.  ``slogdet`` is used so large determinant magnitudes cannot
    overflow merely while extracting their phase.
    """

    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    if initial_samples < 8 or initial_samples & (initial_samples - 1):
        raise ValueError("initial_samples must be a power of two and at least 8")
    if maximum_samples < initial_samples:
        raise ValueError("maximum_samples must be at least initial_samples")
    if maximum_samples & (maximum_samples - 1):
        raise ValueError("maximum_samples must be a power of two")
    if maximum_phase_increment <= 0.0 or maximum_phase_increment >= np.pi:
        raise ValueError("maximum_phase_increment must lie in (0, pi)")

    levels: list[WindingLevel] = []
    previous_winding: int | None = None
    samples = int(initial_samples)

    while samples <= maximum_samples:
        level = _winding_level(operator, center, radius, samples)
        levels.append(level)
        if not level.finite or level.winding_number is None:
            return AdaptiveWindingEvidence(
                converged=False,
                winding_number=None,
                samples_used=samples,
                max_phase_increment=level.max_phase_increment,
                min_log_abs_determinant=level.min_log_abs_determinant,
                message="non-finite or singular determinant sample on contour",
                levels=tuple(levels),
            )
        if (
            previous_winding is not None
            and level.winding_number == previous_winding
            and level.max_phase_increment < maximum_phase_increment
        ):
            return AdaptiveWindingEvidence(
                converged=True,
                winding_number=level.winding_number,
                samples_used=samples,
                max_phase_increment=level.max_phase_increment,
                min_log_abs_determinant=level.min_log_abs_determinant,
                message="two consecutive winding counts agree at resolved phase sampling",
                levels=tuple(levels),
            )
        previous_winding = level.winding_number
        samples *= 2

    final = levels[-1]
    return AdaptiveWindingEvidence(
        converged=False,
        winding_number=final.winding_number,
        samples_used=final.samples,
        max_phase_increment=final.max_phase_increment,
        min_log_abs_determinant=final.min_log_abs_determinant,
        message="maximum contour sampling reached before numerical convergence",
        levels=tuple(levels),
    )



def fast_adjacent_prefilter(
    source_roots: Sequence[complex] | Array,
    original_root_indices: Sequence[int] | Array,
    operators: Mapping[Discretization, LocalHHOperator],
    *,
    source_discretization: Discretization,
    primary_probes: Sequence[Discretization],
    fallback_probe: Discretization,
    stability_tolerance: float,
    root_separation_fraction: float = 0.45,
    pole_separation_fraction: float = 0.45,
    tolerance_radius_fraction: float = 0.95,
    zero_tolerance: float = 1.0e-12,
    initial_samples: int = 16,
    maximum_samples: int = 128,
    maximum_phase_increment: float = np.pi / 2.0,
) -> FastPrefilterResult:
    """Fast adjacent-discretization prefilter with contour fallback.

    One projected Newton center is computed for the source and four nearby
    probes.  Roots that agree immediately are retained.  Roots that agree for
    the forward-J and harmonic probes but not for the backward-J probe receive
    an adaptive local root-count check instead of being rejected by a fragile
    one-step estimate.

    The function is deliberately a *prefilter*.  For maximum confidence,
    verify its survivors with :func:`select_roots_by_local_counts` using
    ``active_positions=result.selected_positions``.
    """

    roots = np.asarray(source_roots, dtype=np.complex128).reshape(-1)
    indices = np.asarray(original_root_indices, dtype=np.int64).reshape(-1)
    if roots.size != indices.size:
        raise ValueError("source_roots and original_root_indices must have equal length")
    if stability_tolerance <= 0.0:
        raise ValueError("stability_tolerance must be positive")
    if not 0.0 < root_separation_fraction < 0.5:
        raise ValueError("root_separation_fraction must lie in (0, 0.5)")
    if not 0.0 < pole_separation_fraction < 0.5:
        raise ValueError("pole_separation_fraction must lie in (0, 0.5)")
    if not 0.0 < tolerance_radius_fraction <= 1.0:
        raise ValueError("tolerance_radius_fraction must lie in (0, 1]")
    primary = tuple(primary_probes)
    required = (source_discretization,) + primary + (fallback_probe,)
    missing = [config for config in required if config not in operators]
    if missing:
        raise ValueError(f"operators are missing required probes: {missing}")

    started = perf_counter()
    centers_by_config: dict[Discretization, tuple[ProjectedNewtonCenter, ...]] = {
        config: tuple(projected_newton_center(operators[config], root) for root in roots)
        for config in dict.fromkeys(required)
    }
    center_arrays = {
        config: np.asarray(
            [item.center if item.finite else np.nan + 1j * np.nan for item in values],
            dtype=np.complex128,
        )
        for config, values in centers_by_config.items()
    }
    source_centers = centers_by_config[source_discretization]
    source_values = center_arrays[source_discretization]
    neighbor_distances = _nearest_neighbor_distances(roots)
    geometry_operators = tuple(operators.values())
    pole_distances = np.asarray(
        [_minimum_declared_pole_distance(root, geometry_operators) for root in roots],
        dtype=float,
    )
    isolation_limits = np.minimum(
        root_separation_fraction * neighbor_distances,
        pole_separation_fraction * pole_distances,
    )
    numerical_floors = 100.0 * np.finfo(float).eps * (1.0 + np.abs(roots))
    zero_mask = np.abs(roots) <= zero_tolerance * (1.0 + np.abs(roots))

    def one_step_diagnostics(probes: Sequence[Discretization]) -> tuple[Array, Array, Array, Array]:
        arrays = [source_values] + [center_arrays[config] for config in probes]
        stacked = np.stack(arrays, axis=1)
        finite = np.all(np.isfinite(stacked.real) & np.isfinite(stacked.imag), axis=1)
        maximum_move = np.zeros(roots.size, dtype=float)
        for config in probes:
            target = center_arrays[config]
            maximum_move = np.maximum(
                maximum_move,
                np.abs(target - source_values)
                / (1.0 + np.maximum(np.abs(target), np.abs(source_values))),
            )
        maximum_source_move = np.max(np.abs(stacked - roots[:, None]), axis=1)
        collapsed = np.isnan(isolation_limits) | (isolation_limits <= numerical_floors)
        leaves_isolation = finite & ~collapsed & (maximum_source_move >= isolation_limits)
        mask = (
            finite
            & ~zero_mask
            & ~collapsed
            & ~leaves_isolation
            & (maximum_move <= stability_tolerance)
        )
        return mask, finite, collapsed, leaves_isolation

    base_mask, _, _, _ = one_step_diagnostics(primary)
    direct_mask, _, _, _ = one_step_diagnostics((fallback_probe,) + primary)
    ambiguous_mask = base_mask & ~direct_mask
    selected_mask = direct_mask.copy()
    status = np.full(roots.size, "prefilter_rejected", dtype=object)
    status[zero_mask] = "zero_or_constraint_root"
    status[base_mask] = "base_candidate"
    status[direct_mask] = "direct_one_step_pass"
    unresolved = 0

    for position in np.flatnonzero(ambiguous_mask):
        center = source_values[position]
        if not _finite_complex(center):
            status[position] = "unresolved_center_correction"
            unresolved += 1
            continue
        neighbor_distance = _distance_to_other_roots(center, roots, int(position))
        pole_distance = _minimum_declared_pole_distance(center, geometry_operators)
        radius = min(
            tolerance_radius_fraction * stability_tolerance * (1.0 + abs(center)),
            root_separation_fraction * neighbor_distance,
            pole_separation_fraction * pole_distance,
        )
        numerical_floor = 100.0 * np.finfo(float).eps * (1.0 + abs(center))
        if not np.isfinite(radius) or radius <= numerical_floor:
            status[position] = "unresolved_collapsed_neighborhood"
            unresolved += 1
            continue
        try:
            evidence = adaptive_winding_number(
                operators[fallback_probe],
                center,
                float(radius),
                initial_samples=initial_samples,
                maximum_samples=maximum_samples,
                maximum_phase_increment=maximum_phase_increment,
            )
        except (FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError, ValueError):
            status[position] = "unresolved_contour_sampling"
            unresolved += 1
            continue
        if not evidence.converged:
            status[position] = "unresolved_contour_sampling"
            unresolved += 1
        elif evidence.winding_number == 1:
            selected_mask[position] = True
            status[position] = "fallback_contour_pass"
        else:
            status[position] = "fallback_contour_reject"

    selected_positions = np.flatnonzero(selected_mask).astype(np.int64)
    return FastPrefilterResult(
        source_discretization=source_discretization,
        primary_probes=primary,
        fallback_probe=fallback_probe,
        stability_tolerance=float(stability_tolerance),
        elapsed_seconds=float(perf_counter() - started),
        base_candidate_positions=np.flatnonzero(base_mask).astype(np.int64),
        direct_pass_positions=np.flatnonzero(direct_mask).astype(np.int64),
        ambiguous_positions=np.flatnonzero(ambiguous_mask).astype(np.int64),
        selected_positions=selected_positions,
        selected_original_indices=indices[selected_positions],
        source_centers=source_centers,
        status_by_position=tuple(str(value) for value in status),
        unresolved_count=int(unresolved),
    )

def select_roots_by_local_counts(
    source_roots: Sequence[complex] | Array,
    original_root_indices: Sequence[int] | Array,
    operators: Mapping[Discretization, LocalHHOperator],
    *,
    source_discretization: Discretization,
    tier_name: str,
    stability_tolerance: float,
    root_separation_fraction: float = 0.45,
    pole_separation_fraction: float = 0.45,
    tolerance_radius_fraction: float = 0.95,
    zero_tolerance: float = 1.0e-12,
    initial_samples: int = 16,
    maximum_samples: int = 128,
    maximum_phase_increment: float = np.pi / 2.0,
    precomputed_centers: Sequence[ProjectedNewtonCenter] | None = None,
    geometry_operators: Sequence[LocalHHOperator] | None = None,
    active_positions: Sequence[int] | Array | None = None,
) -> TierSelectionResult:
    """Select roots whose local disk contains one zero for every operator.

    The same disk is used for every discretization in a tier.  Its radius is
    the minimum of the requested normalized stability tolerance, the local
    source-root separation, and the nearest declared pole over the entire
    tier.  This prevents a root from receiving a different neighborhood in
    each comparison.
    """

    if source_discretization not in operators:
        raise ValueError("source_discretization is missing from operators")
    if stability_tolerance <= 0.0:
        raise ValueError("stability_tolerance must be positive")
    if not 0.0 < root_separation_fraction < 0.5:
        raise ValueError("root_separation_fraction must lie in (0, 0.5)")
    if not 0.0 < pole_separation_fraction < 0.5:
        raise ValueError("pole_separation_fraction must lie in (0, 0.5)")
    if not 0.0 < tolerance_radius_fraction <= 1.0:
        raise ValueError("tolerance_radius_fraction must lie in (0, 1]")

    roots = np.asarray(source_roots, dtype=np.complex128).reshape(-1)
    indices = np.asarray(original_root_indices, dtype=np.int64).reshape(-1)
    if roots.size != indices.size:
        raise ValueError("source_roots and original_root_indices must have equal length")
    if roots.size == 0:
        return TierSelectionResult(
            tier_name=tier_name,
            discretizations=tuple(operators),
            stability_tolerance=float(stability_tolerance),
            elapsed_seconds=0.0,
            selected_positions=np.empty(0, dtype=np.int64),
            selected_original_indices=np.empty(0, dtype=np.int64),
            status_counts={},
            decisions=(),
        )

    configs = tuple(operators.keys())
    source_operator = operators[source_discretization]
    geometry_operator_values = (
        tuple(operators.values())
        if geometry_operators is None
        else tuple(geometry_operators)
    )
    if precomputed_centers is None:
        centers = tuple(projected_newton_center(source_operator, root) for root in roots)
    else:
        centers = tuple(precomputed_centers)
        if len(centers) != roots.size:
            raise ValueError("precomputed_centers must match source_roots")
    if active_positions is None:
        active_mask = np.ones(roots.size, dtype=bool)
    else:
        active_values = np.asarray(active_positions, dtype=np.int64).reshape(-1)
        if np.any(active_values < 0) or np.any(active_values >= roots.size):
            raise ValueError("active_positions contains an out-of-range position")
        active_mask = np.zeros(roots.size, dtype=bool)
        active_mask[active_values] = True

    decisions: list[RootCountDecision] = []
    selected_positions: list[int] = []
    counts: dict[str, int] = {}
    started = perf_counter()

    for position, (root, original_index, center_info) in enumerate(zip(roots, indices, centers)):
        center = center_info.center
        status: str
        radius = 0.0
        failed_config: Discretization | None = None
        evidence_items: list[tuple[Discretization, AdaptiveWindingEvidence]] = []

        if not active_mask[position]:
            status = "not_evaluated_prefilter_reject"
        elif not _finite_complex(root):
            status = "unresolved_nonfinite_source_root"
        elif abs(root) <= zero_tolerance * (1.0 + abs(root)):
            status = "zero_or_constraint_root"
        elif not center_info.finite:
            status = "unresolved_center_correction"
        else:
            source_neighbor_distance = _distance_to_other_roots(root, roots, position)
            center_neighbor_distance = _distance_to_other_roots(center, roots, position)
            source_pole_distance = _minimum_declared_pole_distance(root, (source_operator,))
            center_pole_distance = _minimum_declared_pole_distance(center, geometry_operator_values)
            neighbor_distance = center_neighbor_distance
            pole_distance = center_pole_distance
            numerical_floor = 100.0 * np.finfo(float).eps * (1.0 + abs(root))
            source_geometry_radius = min(
                root_separation_fraction * source_neighbor_distance,
                pole_separation_fraction * source_pole_distance,
            )
            if np.isnan(source_geometry_radius) or source_geometry_radius <= numerical_floor:
                status = "unresolved_collapsed_neighborhood"
            elif abs(center_info.step) >= source_geometry_radius:
                status = "unresolved_center_leaves_isolation"
            else:
                radius = min(
                    tolerance_radius_fraction
                    * stability_tolerance
                    * (1.0 + abs(center)),
                    root_separation_fraction * neighbor_distance,
                    pole_separation_fraction * pole_distance,
                )
                numerical_floor = 100.0 * np.finfo(float).eps * (1.0 + abs(center))
                if not np.isfinite(radius) or radius <= numerical_floor:
                    status = "unresolved_collapsed_neighborhood"
                else:
                    status = "selected"
                    for config, operator in operators.items():
                        try:
                            evidence = adaptive_winding_number(
                                operator,
                                center,
                                float(radius),
                                initial_samples=initial_samples,
                                maximum_samples=maximum_samples,
                                maximum_phase_increment=maximum_phase_increment,
                            )
                        except (FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError, ValueError) as exc:
                            evidence = AdaptiveWindingEvidence(
                                converged=False,
                                winding_number=None,
                                samples_used=0,
                                max_phase_increment=float("inf"),
                                min_log_abs_determinant=float("-inf"),
                                message=str(exc),
                                levels=(),
                            )
                        evidence_items.append((config, evidence))
                        if not evidence.converged:
                            status = "unresolved_contour_sampling"
                            failed_config = config
                            break
                        if evidence.winding_number != 1:
                            status = "rejected_local_root_count"
                            failed_config = config
                            break

        if status == "selected":
            selected_positions.append(position)
        counts[status] = counts.get(status, 0) + 1
        decisions.append(
            RootCountDecision(
                retained_position=position,
                original_root_index=int(original_index),
                source_root=complex(root),
                center=complex(center),
                radius=float(radius),
                status=status,
                first_failed_discretization=failed_config,
                center_evidence=center_info,
                contour_evidence=tuple(evidence_items),
            )
        )

    selected_array = np.asarray(selected_positions, dtype=np.int64)
    return TierSelectionResult(
        tier_name=tier_name,
        discretizations=configs,
        stability_tolerance=float(stability_tolerance),
        elapsed_seconds=float(perf_counter() - started),
        selected_positions=selected_array,
        selected_original_indices=indices[selected_array],
        status_counts=counts,
        decisions=tuple(decisions),
    )



def extend_tier_selection(
    base: TierSelectionResult,
    additional_operators: Mapping[Discretization, LocalHHOperator],
    *,
    tier_name: str,
    initial_samples: int = 16,
    maximum_samples: int = 128,
    maximum_phase_increment: float = np.pi / 2.0,
) -> TierSelectionResult:
    """Extend an already evaluated tier without rescanning rejected roots.

    Only roots selected by ``base`` are tested against the additional
    discretizations.  The center and common disk radius from the base tier are
    preserved, so the result is a genuine nested stress test rather than a new
    neighborhood chosen after seeing the outcome.
    """

    if not additional_operators:
        return TierSelectionResult(
            tier_name=tier_name,
            discretizations=base.discretizations,
            stability_tolerance=base.stability_tolerance,
            elapsed_seconds=0.0,
            selected_positions=base.selected_positions.copy(),
            selected_original_indices=base.selected_original_indices.copy(),
            status_counts=dict(base.status_counts),
            decisions=base.decisions,
        )

    started = perf_counter()
    selected_positions: list[int] = []
    decisions: list[RootCountDecision] = []
    counts: dict[str, int] = {}

    for decision in base.decisions:
        status = decision.status
        failed_config = decision.first_failed_discretization
        evidence_items = list(decision.contour_evidence)

        if decision.status == "selected":
            status = "selected"
            for config, operator in additional_operators.items():
                try:
                    evidence = adaptive_winding_number(
                        operator,
                        decision.center,
                        decision.radius,
                        initial_samples=initial_samples,
                        maximum_samples=maximum_samples,
                        maximum_phase_increment=maximum_phase_increment,
                    )
                except (FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError, ValueError) as exc:
                    evidence = AdaptiveWindingEvidence(
                        converged=False,
                        winding_number=None,
                        samples_used=0,
                        max_phase_increment=float("inf"),
                        min_log_abs_determinant=float("-inf"),
                        message=str(exc),
                        levels=(),
                    )
                evidence_items.append((config, evidence))
                if not evidence.converged:
                    status = "unresolved_core_extension"
                    failed_config = config
                    break
                if evidence.winding_number != 1:
                    status = "rejected_core_extension"
                    failed_config = config
                    break

        if status == "selected":
            selected_positions.append(decision.retained_position)
        counts[status] = counts.get(status, 0) + 1
        decisions.append(
            RootCountDecision(
                retained_position=decision.retained_position,
                original_root_index=decision.original_root_index,
                source_root=decision.source_root,
                center=decision.center,
                radius=decision.radius,
                status=status,
                first_failed_discretization=failed_config,
                center_evidence=decision.center_evidence,
                contour_evidence=tuple(evidence_items),
            )
        )

    selected_array = np.asarray(selected_positions, dtype=np.int64)
    index_by_position = {
        decision.retained_position: decision.original_root_index
        for decision in decisions
    }
    selected_original = np.asarray(
        [index_by_position[int(position)] for position in selected_array],
        dtype=np.int64,
    )
    return TierSelectionResult(
        tier_name=tier_name,
        discretizations=base.discretizations + tuple(additional_operators.keys()),
        stability_tolerance=base.stability_tolerance,
        elapsed_seconds=float(perf_counter() - started),
        selected_positions=selected_array,
        selected_original_indices=selected_original,
        status_counts=counts,
        decisions=tuple(decisions),
    )

def _winding_level(
    operator: LocalHHOperator,
    center: complex,
    radius: float,
    samples: int,
) -> WindingLevel:
    angles = 2.0 * np.pi * np.arange(samples, dtype=float) / samples
    phases = np.empty(samples, dtype=np.complex128)
    log_abs = np.empty(samples, dtype=float)

    try:
        for index, angle in enumerate(angles):
            z = center + radius * np.exp(1j * angle)
            tensor = np.asarray(operator.tensor(z), dtype=np.complex128)
            if not _finite_matrix(tensor):
                raise FloatingPointError("non-finite tensor on contour")
            sign, sample_log_abs = np.linalg.slogdet(tensor)
            sign = complex(sign)
            sign_magnitude = float(abs(sign))
            if (
                not _finite_complex(sign)
                or not np.isfinite(sample_log_abs)
                or sign_magnitude <= np.finfo(float).tiny
            ):
                raise FloatingPointError("singular or non-finite slogdet sample on contour")
            phases[index] = sign / sign_magnitude
            log_abs[index] = float(sample_log_abs)
    except (FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
        return WindingLevel(
            samples=int(samples),
            winding_number=None,
            max_phase_increment=float("inf"),
            min_log_abs_determinant=float("-inf"),
            finite=False,
        )

    increments = np.angle(np.roll(phases, -1) * np.conjugate(phases))
    winding = int(np.rint(float(np.sum(increments)) / (2.0 * np.pi)))
    return WindingLevel(
        samples=int(samples),
        winding_number=winding,
        max_phase_increment=float(np.max(np.abs(increments))),
        min_log_abs_determinant=float(np.min(log_abs)),
        finite=True,
    )


def _vacuum_matrix(operator: HHRationalOperator) -> Array:
    return np.array(
        [
            [
                -operator.ctx.c2 * operator.point.kz**2,
                0.0,
                operator.ctx.c2 * operator.point.kx * operator.point.kz,
            ],
            [
                0.0,
                -operator.ctx.c2 * (operator.point.kx**2 + operator.point.kz**2),
                0.0,
            ],
            [
                operator.ctx.c2 * operator.point.kx * operator.point.kz,
                0.0,
                -operator.ctx.c2 * operator.point.kx**2,
            ],
        ],
        dtype=np.complex128,
    )



def _nearest_neighbor_distances(roots: Array) -> Array:
    values = np.asarray(roots, dtype=np.complex128).reshape(-1)
    if values.size <= 1:
        return np.full(values.size, np.inf, dtype=float)
    distances = np.abs(values[:, None] - values[None, :])
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)

def _distance_to_other_roots(center: complex, roots: Array, position: int) -> float:
    if roots.size <= 1:
        return float("inf")
    distances = np.abs(roots - center)
    distances[position] = np.inf
    finite = distances[np.isfinite(distances)]
    return float(np.min(finite)) if finite.size else float("inf")


def _minimum_declared_pole_distance(
    center: complex,
    operators: Sequence[LocalHHOperator],
) -> float:
    minimum = float("inf")
    for operator in operators:
        poles = np.asarray(operator.declared_poles_normalized, dtype=np.complex128).reshape(-1)
        finite = poles[np.isfinite(poles.real) & np.isfinite(poles.imag)]
        if finite.size:
            minimum = min(minimum, float(np.min(np.abs(finite - center))))
    return minimum


def _finite_matrix(value: Array) -> bool:
    array = np.asarray(value)
    return bool(np.all(np.isfinite(array.real)) and np.all(np.isfinite(array.imag)))


def _finite_complex(value: complex) -> bool:
    z = complex(value)
    return bool(np.isfinite(z.real) and np.isfinite(z.imag))


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]

from __future__ import annotations

"""Shared, family-neutral contracts for BO/PDRK root selection.

The numerical selector implementations are intentionally kept out of this
module.  It contains only immutable protocol, profile, evidence, accounting,
and prediction-manifest models so HH and legacy adapters can share one public
schema without importing validation data or paper-reference code.
"""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


MetadataScalar = str | int | float | bool | None
MetadataItems = tuple[tuple[str, MetadataScalar], ...]


class SelectorFinalState(StrEnum):
    """The only final states admitted for a finite source-spectrum root."""

    DECLARED_AUXILIARY_POLE = "declared_auxiliary_pole"
    CORE_STABLE = "core_stable"
    NEAR_STABLE_ONLY = "near_stable_only"
    UNRESOLVED = "unresolved"
    ZERO_OR_CONSTRAINT_ROOT = "zero_or_constraint_root"
    REJECTED = "rejected"


CANONICAL_SELECTOR_STATES: tuple[SelectorFinalState, ...] = tuple(SelectorFinalState)


@dataclass(frozen=True)
class FrequencyNormalization:
    """Conversion between dimensional frequency and the selector coordinate."""

    scale: float
    reference: str = "wn"
    normalized_units: str = "omega_over_wn"

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("normalization scale must be positive and finite")
        _require_nonempty(self.reference, "normalization reference")
        _require_nonempty(self.normalized_units, "normalized units")

    def normalize(self, dimensional_frequency: complex) -> complex:
        return complex(dimensional_frequency) / self.scale

    def dimensionalize(self, normalized_frequency: complex) -> complex:
        return complex(normalized_frequency) * self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "reference": self.reference,
            "normalized_units": self.normalized_units,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrequencyNormalization:
        return cls(
            scale=float(payload["scale"]),
            reference=str(payload.get("reference", "wn")),
            normalized_units=str(payload.get("normalized_units", "omega_over_wn")),
        )


@dataclass(frozen=True)
class DiscretizationMetadata:
    """Family-neutral description of one local rational approximation."""

    family: str
    pole_order: int | None = None
    harmonic_truncation: int | None = None
    variant: str = ""
    parameters: MetadataItems = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.family, "discretization family")
        if self.pole_order is not None and self.pole_order <= 0:
            raise ValueError("pole_order must be positive when provided")
        if self.harmonic_truncation is not None and self.harmonic_truncation < 0:
            raise ValueError("harmonic_truncation must be non-negative when provided")
        object.__setattr__(self, "parameters", _normalize_metadata(self.parameters))

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.family,
            self.pole_order,
            self.harmonic_truncation,
            self.variant,
            self.parameters,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "pole_order": self.pole_order,
            "harmonic_truncation": self.harmonic_truncation,
            "variant": self.variant,
            "parameters": _metadata_dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DiscretizationMetadata:
        return cls(
            family=str(payload["family"]),
            pole_order=_optional_int(payload.get("pole_order")),
            harmonic_truncation=_optional_int(payload.get("harmonic_truncation")),
            variant=str(payload.get("variant", "")),
            parameters=_normalize_metadata(payload.get("parameters", {})),
        )


@dataclass(frozen=True)
class DeclaredPole:
    """One certified structural pole, expressed in normalized frequency."""

    value_normalized: complex
    multiplicity: int
    provenance: str
    certificate_id: str | None = None

    def __post_init__(self) -> None:
        _require_finite_complex(self.value_normalized, "declared pole")
        if self.multiplicity <= 0:
            raise ValueError("declared-pole multiplicity must be positive")
        _require_nonempty(self.provenance, "declared-pole provenance")
        if self.certificate_id is not None:
            _require_nonempty(self.certificate_id, "declared-pole certificate_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_normalized": _complex_pair(self.value_normalized),
            "multiplicity": int(self.multiplicity),
            "provenance": self.provenance,
            "certificate_id": self.certificate_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeclaredPole:
        return cls(
            value_normalized=_complex_from_pair(payload["value_normalized"]),
            multiplicity=int(payload["multiplicity"]),
            provenance=str(payload["provenance"]),
            certificate_id=_optional_str(payload.get("certificate_id")),
        )


@dataclass(frozen=True)
class LogDeterminant:
    """Overflow-safe determinant magnitude and phase at one local point."""

    log_abs_determinant: float | None
    phase: float | None
    finite: bool

    def __post_init__(self) -> None:
        if self.finite:
            _require_finite_float(self.log_abs_determinant, "log_abs_determinant")
            _require_finite_float(self.phase, "determinant phase")

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_abs_determinant": self.log_abs_determinant,
            "phase": self.phase,
            "finite": bool(self.finite),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LogDeterminant:
        return cls(
            log_abs_determinant=_optional_float(payload.get("log_abs_determinant")),
            phase=_optional_float(payload.get("phase")),
            finite=bool(payload["finite"]),
        )


@runtime_checkable
class LocalRationalOperator(Protocol):
    """Common numerical surface implemented by HH and legacy adapters.

    Public roots passed to :meth:`tensor` and :meth:`slogdet`, and all declared
    poles, use ``normalization.normalized_units``.
    """

    @property
    def normalization(self) -> FrequencyNormalization:
        ...

    @property
    def discretization(self) -> DiscretizationMetadata:
        ...

    @property
    def declared_poles(self) -> tuple[DeclaredPole, ...]:
        ...

    def tensor(self, root_normalized: complex) -> NDArray[np.complex128]:
        ...

    def slogdet(self, root_normalized: complex) -> LogDeterminant:
        ...


@dataclass(frozen=True)
class SourceProfile:
    discretization: DiscretizationMetadata
    operator_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "discretization": self.discretization.to_dict(),
            "operator_id": self.operator_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceProfile:
        return cls(
            discretization=DiscretizationMetadata.from_dict(payload["discretization"]),
            operator_id=str(payload.get("operator_id", "")),
        )


@dataclass(frozen=True)
class ProbeTierProfile:
    discretizations: tuple[DiscretizationMetadata, ...]
    require_all: bool = True

    def __post_init__(self) -> None:
        values = tuple(self.discretizations)
        if not values:
            raise ValueError("a probe tier must contain at least one discretization")
        _require_unique_discretizations(values, "probe tier")
        object.__setattr__(self, "discretizations", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discretizations": [item.to_dict() for item in self.discretizations],
            "require_all": bool(self.require_all),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeTierProfile:
        return cls(
            discretizations=tuple(
                DiscretizationMetadata.from_dict(item)
                for item in payload.get("discretizations", ())
            ),
            require_all=bool(payload.get("require_all", True)),
        )


@dataclass(frozen=True)
class FallbackProbeProfile:
    discretization: DiscretizationMetadata
    trigger_statuses: tuple[str, ...] = ("ambiguous",)

    def __post_init__(self) -> None:
        statuses = tuple(str(value) for value in self.trigger_statuses)
        if not statuses or any(not value for value in statuses):
            raise ValueError("fallback trigger_statuses must be non-empty strings")
        if len(statuses) != len(set(statuses)):
            raise ValueError("fallback trigger_statuses must be unique")
        object.__setattr__(self, "trigger_statuses", statuses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discretization": self.discretization.to_dict(),
            "trigger_statuses": list(self.trigger_statuses),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FallbackProbeProfile:
        return cls(
            discretization=DiscretizationMetadata.from_dict(payload["discretization"]),
            trigger_statuses=tuple(str(value) for value in payload.get("trigger_statuses", ())),
        )


@dataclass(frozen=True)
class ContourProfile:
    sample_levels: tuple[int, ...] = (16, 32, 64, 128)
    maximum_phase_increment: float = math.pi / 2.0
    consecutive_equal_counts: int = 2
    required_winding_number: int = 1
    rigorous: bool = False

    def __post_init__(self) -> None:
        levels = tuple(int(value) for value in self.sample_levels)
        if len(levels) < self.consecutive_equal_counts:
            raise ValueError("sample_levels cannot establish the requested consecutive counts")
        if any(value < 8 or value & (value - 1) for value in levels):
            raise ValueError("contour sample levels must be powers of two and at least 8")
        if any(right <= left for left, right in zip(levels, levels[1:], strict=False)):
            raise ValueError("contour sample levels must be strictly increasing")
        if not 0.0 < self.maximum_phase_increment < math.pi:
            raise ValueError("maximum_phase_increment must lie in (0, pi)")
        if self.consecutive_equal_counts < 2:
            raise ValueError("at least two consecutive equal winding counts are required")
        if self.rigorous:
            raise ValueError("adaptive contour evidence must remain rigorous=False")
        object.__setattr__(self, "sample_levels", levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_levels": list(self.sample_levels),
            "maximum_phase_increment": float(self.maximum_phase_increment),
            "consecutive_equal_counts": int(self.consecutive_equal_counts),
            "required_winding_number": int(self.required_winding_number),
            "rigorous": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContourProfile:
        return cls(
            sample_levels=tuple(int(value) for value in payload.get("sample_levels", ())),
            maximum_phase_increment=float(payload["maximum_phase_increment"]),
            consecutive_equal_counts=int(payload.get("consecutive_equal_counts", 2)),
            required_winding_number=int(payload.get("required_winding_number", 1)),
            rigorous=bool(payload.get("rigorous", False)),
        )


@dataclass(frozen=True)
class ToleranceProfile:
    normalized_movement_epsilon: float
    zero_root_tolerance: float = 1.0e-12
    structural_pole_match_tolerance: float = 1.0e-8
    root_separation_fraction: float = 0.45
    pole_separation_fraction: float = 0.45
    tolerance_radius_fraction: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "normalized_movement_epsilon",
            "zero_root_tolerance",
            "structural_pole_match_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "root_separation_fraction",
            "pole_separation_fraction",
            "tolerance_radius_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")

    def normalized_movement(self, source_root: complex, target_root: complex) -> float:
        return float(abs(complex(target_root) - complex(source_root)) / (1.0 + abs(source_root)))

    def accepts(self, source_root: complex, target_root: complex) -> bool:
        return self.normalized_movement(source_root, target_root) <= self.normalized_movement_epsilon

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_movement_epsilon": float(self.normalized_movement_epsilon),
            "normalized_movement_formula": "abs(delta_omega)/(1+abs(omega))",
            "zero_root_tolerance": float(self.zero_root_tolerance),
            "structural_pole_match_tolerance": float(self.structural_pole_match_tolerance),
            "root_separation_fraction": float(self.root_separation_fraction),
            "pole_separation_fraction": float(self.pole_separation_fraction),
            "tolerance_radius_fraction": float(self.tolerance_radius_fraction),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ToleranceProfile:
        return cls(
            normalized_movement_epsilon=float(payload["normalized_movement_epsilon"]),
            zero_root_tolerance=float(payload.get("zero_root_tolerance", 1.0e-12)),
            structural_pole_match_tolerance=float(
                payload.get("structural_pole_match_tolerance", 1.0e-8)
            ),
            root_separation_fraction=float(payload.get("root_separation_fraction", 0.45)),
            pole_separation_fraction=float(payload.get("pole_separation_fraction", 0.45)),
            tolerance_radius_fraction=float(payload.get("tolerance_radius_fraction", 0.95)),
        )


@dataclass(frozen=True)
class ScanScope:
    case_id: str
    scan_indices: tuple[int, ...] | None = None
    source_variant: str = "raw"
    partition: str = "production"
    held_out: bool = False
    expected_scan_count: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.case_id, "scan-scope case_id")
        _require_nonempty(self.source_variant, "scan-scope source_variant")
        _require_nonempty(self.partition, "scan-scope partition")
        if self.expected_scan_count is not None and self.expected_scan_count <= 0:
            raise ValueError("expected_scan_count must be positive when provided")
        if self.scan_indices is not None:
            indices = tuple(int(value) for value in self.scan_indices)
            if not indices:
                raise ValueError("scan_indices must be None for all scans or a non-empty tuple")
            if any(value < 0 for value in indices):
                raise ValueError("scan indices must be non-negative")
            if len(indices) != len(set(indices)):
                raise ValueError("scan indices must be unique")
            if indices != tuple(sorted(indices)):
                raise ValueError("scan indices must be sorted")
            if self.expected_scan_count is not None and any(
                value >= self.expected_scan_count for value in indices
            ):
                raise ValueError("scan index is outside expected_scan_count")
            object.__setattr__(self, "scan_indices", indices)

    def contains(self, scan_index: int) -> bool:
        return self.scan_indices is None or int(scan_index) in self.scan_indices

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scan_indices_0based": None if self.scan_indices is None else list(self.scan_indices),
            "source_variant": self.source_variant,
            "partition": self.partition,
            "held_out": bool(self.held_out),
            "expected_scan_count": self.expected_scan_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScanScope:
        values = payload.get("scan_indices_0based")
        return cls(
            case_id=str(payload["case_id"]),
            scan_indices=None if values is None else tuple(int(value) for value in values),
            source_variant=str(payload.get("source_variant", "raw")),
            partition=str(payload.get("partition", "production")),
            held_out=bool(payload.get("held_out", False)),
            expected_scan_count=_optional_int(payload.get("expected_scan_count")),
        )


@dataclass(frozen=True)
class SelectorProfile:
    profile_id: str
    family: str
    source: SourceProfile
    near: ProbeTierProfile
    fallback: FallbackProbeProfile | None
    core: ProbeTierProfile | None
    contour: ContourProfile
    tolerance: ToleranceProfile
    scan_scope: ScanScope
    provenance: str
    metadata: MetadataItems = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_nonempty(self.profile_id, "selector profile_id")
        _require_nonempty(self.family, "selector family")
        _require_nonempty(self.provenance, "selector profile provenance")
        if self.schema_version != 1:
            raise ValueError("unsupported selector profile schema_version")
        discretizations = [self.source.discretization, *self.near.discretizations]
        if self.fallback is not None:
            discretizations.append(self.fallback.discretization)
        if self.core is not None:
            discretizations.extend(self.core.discretizations)
        inconsistent = [item.family for item in discretizations if item.family != self.family]
        if inconsistent:
            raise ValueError("all profile discretizations must use the declared family")
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "family": self.family,
            "source": self.source.to_dict(),
            "near": self.near.to_dict(),
            "fallback": None if self.fallback is None else self.fallback.to_dict(),
            "core": None if self.core is None else self.core.to_dict(),
            "contour": self.contour.to_dict(),
            "tolerance": self.tolerance.to_dict(),
            "scan_scope": self.scan_scope.to_dict(),
            "provenance": self.provenance,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SelectorProfile:
        fallback = payload.get("fallback")
        core = payload.get("core")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            profile_id=str(payload["profile_id"]),
            family=str(payload["family"]),
            source=SourceProfile.from_dict(payload["source"]),
            near=ProbeTierProfile.from_dict(payload["near"]),
            fallback=None if fallback is None else FallbackProbeProfile.from_dict(fallback),
            core=None if core is None else ProbeTierProfile.from_dict(core),
            contour=ContourProfile.from_dict(payload["contour"]),
            tolerance=ToleranceProfile.from_dict(payload["tolerance"]),
            scan_scope=ScanScope.from_dict(payload["scan_scope"]),
            provenance=str(payload["provenance"]),
            metadata=_normalize_metadata(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class StructuralPoleEvidence:
    matched: bool
    declared_pole: DeclaredPole | None = None
    matching_distance: float | None = None
    multiplicity_slot: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.matched:
            if self.declared_pole is None:
                raise ValueError("matched structural-pole evidence requires a declared pole")
            _require_nonnegative_float(self.matching_distance, "pole matching_distance")
            if self.multiplicity_slot is None or not 0 <= self.multiplicity_slot < self.declared_pole.multiplicity:
                raise ValueError("multiplicity_slot must identify a certified pole occurrence")
        elif self.multiplicity_slot is not None:
            raise ValueError("unmatched pole evidence cannot have a multiplicity_slot")
        if self.matching_distance is not None:
            _require_nonnegative_float(self.matching_distance, "pole matching_distance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": bool(self.matched),
            "declared_pole": None if self.declared_pole is None else self.declared_pole.to_dict(),
            "matching_distance": self.matching_distance,
            "multiplicity_slot": self.multiplicity_slot,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StructuralPoleEvidence:
        pole = payload.get("declared_pole")
        return cls(
            matched=bool(payload["matched"]),
            declared_pole=None if pole is None else DeclaredPole.from_dict(pole),
            matching_distance=_optional_float(payload.get("matching_distance")),
            multiplicity_slot=_optional_int(payload.get("multiplicity_slot")),
            message=str(payload.get("message", "")),
        )


@dataclass(frozen=True)
class ContourLevelEvidence:
    samples: int
    winding_number: int | None
    max_phase_increment: float | None
    min_log_abs_determinant: float | None
    finite: bool

    def __post_init__(self) -> None:
        if self.samples < 8:
            raise ValueError("contour evidence samples must be at least 8")
        if self.max_phase_increment is not None:
            _require_nonnegative_float(self.max_phase_increment, "max_phase_increment")
        if self.min_log_abs_determinant is not None:
            _require_finite_float(self.min_log_abs_determinant, "min_log_abs_determinant")
        if self.finite and (
            self.winding_number is None
            or self.max_phase_increment is None
            or self.min_log_abs_determinant is None
        ):
            raise ValueError("finite contour evidence requires count, phase, and log magnitude")

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": int(self.samples),
            "winding_number": self.winding_number,
            "max_phase_increment": self.max_phase_increment,
            "min_log_abs_determinant": self.min_log_abs_determinant,
            "finite": bool(self.finite),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContourLevelEvidence:
        return cls(
            samples=int(payload["samples"]),
            winding_number=_optional_int(payload.get("winding_number")),
            max_phase_increment=_optional_float(payload.get("max_phase_increment")),
            min_log_abs_determinant=_optional_float(payload.get("min_log_abs_determinant")),
            finite=bool(payload["finite"]),
        )


@dataclass(frozen=True)
class ProbeEvidence:
    discretization: DiscretizationMetadata
    status: str
    predicted_root: complex | None = None
    normalized_movement: float | None = None
    winding_number: int | None = None
    contour_levels: tuple[ContourLevelEvidence, ...] = ()
    message: str = ""
    rigorous: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.status, "probe status")
        if self.predicted_root is not None:
            _require_finite_complex(self.predicted_root, "predicted root")
        if self.normalized_movement is not None:
            _require_nonnegative_float(self.normalized_movement, "normalized movement")
        if self.rigorous:
            raise ValueError("probe evidence must remain rigorous=False")
        levels = tuple(self.contour_levels)
        if any(right.samples <= left.samples for left, right in zip(levels, levels[1:], strict=False)):
            raise ValueError("contour evidence levels must have increasing sample counts")
        object.__setattr__(self, "contour_levels", levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discretization": self.discretization.to_dict(),
            "status": self.status,
            "predicted_root": None if self.predicted_root is None else _complex_pair(self.predicted_root),
            "normalized_movement": self.normalized_movement,
            "winding_number": self.winding_number,
            "contour_levels": [item.to_dict() for item in self.contour_levels],
            "message": self.message,
            "rigorous": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeEvidence:
        predicted = payload.get("predicted_root")
        return cls(
            discretization=DiscretizationMetadata.from_dict(payload["discretization"]),
            status=str(payload["status"]),
            predicted_root=None if predicted is None else _complex_from_pair(predicted),
            normalized_movement=_optional_float(payload.get("normalized_movement")),
            winding_number=_optional_int(payload.get("winding_number")),
            contour_levels=tuple(
                ContourLevelEvidence.from_dict(item) for item in payload.get("contour_levels", ())
            ),
            message=str(payload.get("message", "")),
            rigorous=bool(payload.get("rigorous", False)),
        )


@dataclass(frozen=True)
class RootSelectionEvidence:
    raw_index: int
    source_root: complex
    final_state: SelectorFinalState
    stability_tolerance: float
    retained_position: int | None = None
    structural_pole: StructuralPoleEvidence | None = None
    corrected_center: complex | None = None
    isolation_radius: float | None = None
    probes: tuple[ProbeEvidence, ...] = ()
    failed_probe: DiscretizationMetadata | None = None
    stage_statuses: tuple[tuple[str, str], ...] = ()
    rigorous: bool = False

    def __post_init__(self) -> None:
        if self.raw_index < 0:
            raise ValueError("raw_index must be non-negative")
        _require_finite_complex(self.source_root, "source root")
        object.__setattr__(self, "final_state", SelectorFinalState(self.final_state))
        if not math.isfinite(self.stability_tolerance) or self.stability_tolerance <= 0.0:
            raise ValueError("stability_tolerance must be positive and finite")
        if self.retained_position is not None and self.retained_position < 0:
            raise ValueError("retained_position must be non-negative")
        if self.corrected_center is not None:
            _require_finite_complex(self.corrected_center, "corrected center")
        if self.isolation_radius is not None:
            _require_nonnegative_float(self.isolation_radius, "isolation radius")
        if self.rigorous:
            raise ValueError("root evidence must remain rigorous=False")
        probes = tuple(self.probes)
        _require_unique_discretizations(
            tuple(item.discretization for item in probes), "per-root probe evidence"
        )
        object.__setattr__(self, "probes", probes)
        statuses = _normalize_string_items(self.stage_statuses)
        object.__setattr__(self, "stage_statuses", statuses)
        is_pole = self.final_state is SelectorFinalState.DECLARED_AUXILIARY_POLE
        pole_matched = self.structural_pole is not None and self.structural_pole.matched
        if is_pole != pole_matched:
            raise ValueError(
                "declared_auxiliary_pole state must correspond exactly to matched structural-pole evidence"
            )
        if is_pole and self.retained_position is not None:
            raise ValueError("a declared auxiliary pole cannot have a retained_position")
        if self.failed_probe is not None and self.failed_probe.key not in {
            item.discretization.key for item in probes
        }:
            raise ValueError("failed_probe must identify one of the recorded probes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_index_0based": int(self.raw_index),
            "retained_position_0based": self.retained_position,
            "source_root": _complex_pair(self.source_root),
            "final_state": self.final_state.value,
            "structural_pole_evidence": (
                None if self.structural_pole is None else self.structural_pole.to_dict()
            ),
            "corrected_center": (
                None if self.corrected_center is None else _complex_pair(self.corrected_center)
            ),
            "isolation_radius": self.isolation_radius,
            "probe_evidence": [item.to_dict() for item in self.probes],
            "failed_probe": None if self.failed_probe is None else self.failed_probe.to_dict(),
            "stability_tolerance": float(self.stability_tolerance),
            "stage_statuses": dict(self.stage_statuses),
            "rigorous": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RootSelectionEvidence:
        pole = payload.get("structural_pole_evidence")
        center = payload.get("corrected_center")
        failed = payload.get("failed_probe")
        return cls(
            raw_index=int(payload["raw_index_0based"]),
            retained_position=_optional_int(payload.get("retained_position_0based")),
            source_root=_complex_from_pair(payload["source_root"]),
            final_state=SelectorFinalState(str(payload["final_state"])),
            structural_pole=None if pole is None else StructuralPoleEvidence.from_dict(pole),
            corrected_center=None if center is None else _complex_from_pair(center),
            isolation_radius=_optional_float(payload.get("isolation_radius")),
            probes=tuple(ProbeEvidence.from_dict(item) for item in payload.get("probe_evidence", ())),
            failed_probe=None if failed is None else DiscretizationMetadata.from_dict(failed),
            stability_tolerance=float(payload["stability_tolerance"]),
            stage_statuses=_normalize_string_items(payload.get("stage_statuses", {})),
            rigorous=bool(payload.get("rigorous", False)),
        )


@dataclass(frozen=True)
class TierSelectionResult:
    tier_name: str
    discretizations: tuple[DiscretizationMetadata, ...]
    selected_raw_indices: tuple[int, ...]
    status_counts: tuple[tuple[str, int], ...]
    elapsed_seconds: float
    rigorous: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.tier_name, "tier_name")
        discretizations = tuple(self.discretizations)
        _require_unique_discretizations(discretizations, "tier result")
        indices = tuple(int(value) for value in self.selected_raw_indices)
        if any(value < 0 for value in indices):
            raise ValueError("selected raw indices must be non-negative")
        if indices != tuple(sorted(set(indices))):
            raise ValueError("selected raw indices must be sorted and unique")
        counts = _normalize_count_items(self.status_counts)
        _require_nonnegative_float(self.elapsed_seconds, "tier elapsed_seconds")
        if self.rigorous:
            raise ValueError("tier evidence must remain rigorous=False")
        object.__setattr__(self, "discretizations", discretizations)
        object.__setattr__(self, "selected_raw_indices", indices)
        object.__setattr__(self, "status_counts", counts)

    @property
    def selected_index_sha256(self) -> str:
        return _index_hash(self.selected_raw_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "discretizations": [item.to_dict() for item in self.discretizations],
            "selected_count": len(self.selected_raw_indices),
            "selected_raw_indices_0based": list(self.selected_raw_indices),
            "selected_index_sha256": self.selected_index_sha256,
            "status_counts": dict(self.status_counts),
            "elapsed_seconds": float(self.elapsed_seconds),
            "rigorous": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TierSelectionResult:
        result = cls(
            tier_name=str(payload["tier_name"]),
            discretizations=tuple(
                DiscretizationMetadata.from_dict(item)
                for item in payload.get("discretizations", ())
            ),
            selected_raw_indices=tuple(
                int(value) for value in payload.get("selected_raw_indices_0based", ())
            ),
            status_counts=_normalize_count_items(payload.get("status_counts", {})),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            rigorous=bool(payload.get("rigorous", False)),
        )
        expected_hash = payload.get("selected_index_sha256")
        if expected_hash is not None and str(expected_hash) != result.selected_index_sha256:
            raise ValueError("tier selected-index hash mismatch")
        return result


@dataclass(frozen=True)
class SpectrumAccounting:
    source_root_count: int
    state_counts: tuple[tuple[SelectorFinalState, int], ...]

    def __post_init__(self) -> None:
        if self.source_root_count < 0:
            raise ValueError("source_root_count must be non-negative")
        normalized = tuple((SelectorFinalState(state), int(count)) for state, count in self.state_counts)
        states = tuple(state for state, _ in normalized)
        if len(states) != len(set(states)):
            raise ValueError("selector state accounting contains duplicate states")
        if set(states) != set(CANONICAL_SELECTOR_STATES):
            missing = sorted(state.value for state in set(CANONICAL_SELECTOR_STATES) - set(states))
            extra = sorted(str(state) for state in set(states) - set(CANONICAL_SELECTOR_STATES))
            raise ValueError(f"selector accounting must contain exactly six states; missing={missing}, extra={extra}")
        counts = dict(normalized)
        if any(value < 0 for value in counts.values()):
            raise ValueError("selector state counts must be non-negative")
        if sum(counts.values()) != self.source_root_count:
            raise ValueError(
                "selector state counts do not sum to source_root_count: "
                f"{sum(counts.values())} != {self.source_root_count}"
            )
        canonical = tuple((state, counts[state]) for state in CANONICAL_SELECTOR_STATES)
        object.__setattr__(self, "state_counts", canonical)

    @classmethod
    def from_evidence(
        cls,
        source_root_count: int,
        evidence: Sequence[RootSelectionEvidence],
    ) -> SpectrumAccounting:
        counts = Counter(item.final_state for item in evidence)
        return cls(
            source_root_count=int(source_root_count),
            state_counts=tuple((state, int(counts.get(state, 0))) for state in CANONICAL_SELECTOR_STATES),
        )

    def count(self, state: SelectorFinalState | str) -> int:
        return dict(self.state_counts)[SelectorFinalState(state)]

    @property
    def retained_nonpole_count(self) -> int:
        return self.source_root_count - self.count(SelectorFinalState.DECLARED_AUXILIARY_POLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_root_count": int(self.source_root_count),
            "state_counts": {state.value: int(count) for state, count in self.state_counts},
            "state_count_sum": int(sum(count for _, count in self.state_counts)),
            "retained_nonpole_count": int(self.retained_nonpole_count),
            "accounting_valid": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpectrumAccounting:
        counts = payload["state_counts"]
        result = cls(
            source_root_count=int(payload["source_root_count"]),
            state_counts=tuple((SelectorFinalState(key), int(value)) for key, value in counts.items()),
        )
        if "state_count_sum" in payload and int(payload["state_count_sum"]) != result.source_root_count:
            raise ValueError("serialized selector state_count_sum is inconsistent")
        if payload.get("accounting_valid", True) is not True:
            raise ValueError("serialized selector accounting is not marked valid")
        return result


@dataclass(frozen=True)
class SelectorTiming:
    source_eigensolve_seconds: float
    operator_build_seconds: float
    prefilter_seconds: float
    near_verification_seconds: float
    core_extension_seconds: float
    prediction_serialization_seconds: float
    selector_total_seconds: float
    end_to_end_seconds: float
    metadata: MetadataItems = ()

    def __post_init__(self) -> None:
        for name in (
            "source_eigensolve_seconds",
            "operator_build_seconds",
            "prefilter_seconds",
            "near_verification_seconds",
            "core_extension_seconds",
            "prediction_serialization_seconds",
            "selector_total_seconds",
            "end_to_end_seconds",
        ):
            _require_nonnegative_float(getattr(self, name), name)
        if self.end_to_end_seconds + 1.0e-15 < self.source_eigensolve_seconds:
            raise ValueError("end_to_end_seconds cannot be shorter than source eigensolve")
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_eigensolve_seconds": float(self.source_eigensolve_seconds),
            "operator_build_seconds": float(self.operator_build_seconds),
            "prefilter_seconds": float(self.prefilter_seconds),
            "near_verification_seconds": float(self.near_verification_seconds),
            "core_extension_seconds": float(self.core_extension_seconds),
            "prediction_serialization_seconds": float(self.prediction_serialization_seconds),
            "selector_total_seconds": float(self.selector_total_seconds),
            "end_to_end_seconds": float(self.end_to_end_seconds),
            "oracle_evaluation_included": False,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SelectorTiming:
        if payload.get("oracle_evaluation_included", False):
            raise ValueError("selector timing cannot include oracle evaluation")
        return cls(
            source_eigensolve_seconds=float(payload["source_eigensolve_seconds"]),
            operator_build_seconds=float(payload["operator_build_seconds"]),
            prefilter_seconds=float(payload["prefilter_seconds"]),
            near_verification_seconds=float(payload["near_verification_seconds"]),
            core_extension_seconds=float(payload["core_extension_seconds"]),
            prediction_serialization_seconds=float(payload["prediction_serialization_seconds"]),
            selector_total_seconds=float(payload["selector_total_seconds"]),
            end_to_end_seconds=float(payload["end_to_end_seconds"]),
            metadata=_normalize_metadata(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SourceSpectrumProvenance:
    case_id: str
    scan_index: int
    root_count: int
    spectrum_sha256: str
    solver_id: str
    artifact_path: str = ""
    normalized_units: str = "omega_over_wn"
    metadata: MetadataItems = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.case_id, "source case_id")
        if self.scan_index < 0:
            raise ValueError("source scan_index must be non-negative")
        if self.root_count < 0:
            raise ValueError("source root_count must be non-negative")
        _require_sha256(self.spectrum_sha256, "source spectrum_sha256")
        _require_nonempty(self.solver_id, "source solver_id")
        _require_nonempty(self.normalized_units, "source normalized_units")
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scan_index_0based": int(self.scan_index),
            "root_count": int(self.root_count),
            "spectrum_sha256": self.spectrum_sha256,
            "solver_id": self.solver_id,
            "artifact_path": self.artifact_path,
            "normalized_units": self.normalized_units,
            "metadata": _metadata_dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceSpectrumProvenance:
        return cls(
            case_id=str(payload["case_id"]),
            scan_index=int(payload["scan_index_0based"]),
            root_count=int(payload["root_count"]),
            spectrum_sha256=str(payload["spectrum_sha256"]),
            solver_id=str(payload["solver_id"]),
            artifact_path=str(payload.get("artifact_path", "")),
            normalized_units=str(payload.get("normalized_units", "omega_over_wn")),
            metadata=_normalize_metadata(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PredictionManifest:
    selector_id: str
    generated_at_utc: str
    source: SourceSpectrumProvenance
    profile: SelectorProfile
    roots: tuple[RootSelectionEvidence, ...]
    tier_results: tuple[TierSelectionResult, ...]
    accounting: SpectrumAccounting
    timings: SelectorTiming
    numerical_failure_counts: tuple[tuple[str, int], ...] = ()
    family_metadata: MetadataItems = ()
    selector_input_manifest_sha256: str | None = None
    ground_truth_read_during_prediction: bool = False
    rigorous: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_nonempty(self.selector_id, "selector_id")
        _require_nonempty(self.generated_at_utc, "generated_at_utc")
        if self.schema_version != 1:
            raise ValueError("unsupported prediction manifest schema_version")
        if self.source.case_id != self.profile.scan_scope.case_id:
            raise ValueError("source case_id differs from selector profile scan scope")
        if not self.profile.scan_scope.contains(self.source.scan_index):
            raise ValueError("source scan_index is outside the selector profile scope")
        if self.source.root_count != self.accounting.source_root_count:
            raise ValueError("source provenance and spectrum accounting root counts differ")
        if self.selector_input_manifest_sha256 is not None:
            _require_sha256(
                self.selector_input_manifest_sha256, "selector_input_manifest_sha256"
            )
        if self.ground_truth_read_during_prediction:
            raise ValueError("prediction manifests cannot admit ground-truth reads")
        if self.rigorous:
            raise ValueError("selector prediction manifests must remain rigorous=False")
        roots = tuple(self.roots)
        tiers = tuple(self.tier_results)
        if len({item.tier_name for item in tiers}) != len(tiers):
            raise ValueError("prediction manifest contains duplicate tier names")
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "tier_results", tiers)
        object.__setattr__(
            self,
            "numerical_failure_counts",
            _normalize_count_items(self.numerical_failure_counts),
        )
        object.__setattr__(self, "family_metadata", _normalize_metadata(self.family_metadata))
        validate_root_evidence_accounting(roots, self.accounting)
        root_indices = {item.raw_index for item in roots}
        for tier in tiers:
            invalid = sorted(set(tier.selected_raw_indices) - root_indices)
            if invalid:
                raise ValueError(
                    f"tier {tier.tier_name!r} selects indices absent from root evidence: {invalid}"
                )

    @property
    def selected_index_hashes(self) -> dict[str, dict[str, Any]]:
        groups: dict[str, tuple[int, ...]] = {}
        for state in CANONICAL_SELECTOR_STATES:
            groups[state.value] = tuple(
                sorted(item.raw_index for item in self.roots if item.final_state is state)
            )
        groups["near_stable"] = tuple(
            sorted(
                item.raw_index
                for item in self.roots
                if item.final_state
                in {SelectorFinalState.NEAR_STABLE_ONLY, SelectorFinalState.CORE_STABLE}
            )
        )
        return {
            name: {"count": len(indices), "sha256": _index_hash(indices)}
            for name, indices in groups.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": "bo_selector_prediction_manifest_v1",
            "selector_id": self.selector_id,
            "generated_at_utc": self.generated_at_utc,
            "source": self.source.to_dict(),
            "profile": self.profile.to_dict(),
            "per_root_evidence": [item.to_dict() for item in self.roots],
            "tier_results": [item.to_dict() for item in self.tier_results],
            "spectrum_accounting": self.accounting.to_dict(),
            "timings_seconds": self.timings.to_dict(),
            "selected_index_hashes": self.selected_index_hashes,
            "numerical_failure_counts": dict(self.numerical_failure_counts),
            "family_metadata": _metadata_dict(self.family_metadata),
            "selector_input_manifest_sha256": self.selector_input_manifest_sha256,
            "information_barrier": {
                "ground_truth_read_during_prediction": False,
                "prediction_is_immutable": True,
            },
            "rigorous": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PredictionManifest:
        if payload.get("contract", "bo_selector_prediction_manifest_v1") != "bo_selector_prediction_manifest_v1":
            raise ValueError("unsupported prediction manifest contract")
        barrier = payload.get("information_barrier", {})
        manifest = cls(
            schema_version=int(payload.get("schema_version", 1)),
            selector_id=str(payload["selector_id"]),
            generated_at_utc=str(payload["generated_at_utc"]),
            source=SourceSpectrumProvenance.from_dict(payload["source"]),
            profile=SelectorProfile.from_dict(payload["profile"]),
            roots=tuple(
                RootSelectionEvidence.from_dict(item)
                for item in payload.get("per_root_evidence", ())
            ),
            tier_results=tuple(
                TierSelectionResult.from_dict(item) for item in payload.get("tier_results", ())
            ),
            accounting=SpectrumAccounting.from_dict(payload["spectrum_accounting"]),
            timings=SelectorTiming.from_dict(payload["timings_seconds"]),
            numerical_failure_counts=_normalize_count_items(
                payload.get("numerical_failure_counts", {})
            ),
            family_metadata=_normalize_metadata(payload.get("family_metadata", {})),
            selector_input_manifest_sha256=_optional_str(
                payload.get("selector_input_manifest_sha256")
            ),
            ground_truth_read_during_prediction=bool(
                barrier.get("ground_truth_read_during_prediction", False)
            ),
            rigorous=bool(payload.get("rigorous", False)),
        )
        serialized_hashes = payload.get("selected_index_hashes")
        if serialized_hashes is not None and serialized_hashes != manifest.selected_index_hashes:
            raise ValueError("prediction selected-index hashes do not match per-root evidence")
        return manifest

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def validate_root_evidence_accounting(
    evidence: Sequence[RootSelectionEvidence],
    accounting: SpectrumAccounting,
) -> None:
    """Validate one immutable record and one final state for every source root."""

    roots = tuple(evidence)
    if len(roots) != accounting.source_root_count:
        raise ValueError(
            "per-root evidence count differs from source_root_count: "
            f"{len(roots)} != {accounting.source_root_count}"
        )
    indices = [item.raw_index for item in roots]
    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate raw root indices in selector evidence: {duplicates}")
    expected = set(range(accounting.source_root_count))
    actual = set(indices)
    if actual != expected:
        raise ValueError(
            "selector evidence must cover every source root exactly once; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    computed = SpectrumAccounting.from_evidence(accounting.source_root_count, roots)
    if computed.state_counts != accounting.state_counts:
        raise ValueError(
            "per-root final states disagree with aggregate spectrum accounting: "
            f"computed={computed.to_dict()['state_counts']}, "
            f"declared={accounting.to_dict()['state_counts']}"
        )


def _normalize_metadata(
    values: Mapping[str, MetadataScalar] | Sequence[tuple[str, MetadataScalar]],
) -> MetadataItems:
    items = values.items() if isinstance(values, Mapping) else values
    normalized: list[tuple[str, MetadataScalar]] = []
    for raw_key, value in items:
        key = str(raw_key)
        _require_nonempty(key, "metadata key")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise TypeError(f"metadata value for {key!r} must be a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metadata value for {key!r} must be finite")
        normalized.append((key, value))
    normalized.sort(key=lambda item: item[0])
    if len(normalized) != len({key for key, _ in normalized}):
        raise ValueError("metadata keys must be unique")
    return tuple(normalized)


def _metadata_dict(values: MetadataItems) -> dict[str, MetadataScalar]:
    return {key: value for key, value in values}


def _normalize_string_items(
    values: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized = sorted((str(key), str(value)) for key, value in items)
    if any(not key or not value for key, value in normalized):
        raise ValueError("status keys and values must be non-empty")
    if len(normalized) != len({key for key, _ in normalized}):
        raise ValueError("status keys must be unique")
    return tuple(normalized)


def _normalize_count_items(
    values: Mapping[str, int] | Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized = sorted((str(key), int(value)) for key, value in items)
    if any(not key for key, _ in normalized):
        raise ValueError("count keys must be non-empty")
    if any(value < 0 for _, value in normalized):
        raise ValueError("counts must be non-negative")
    if len(normalized) != len({key for key, _ in normalized}):
        raise ValueError("count keys must be unique")
    return tuple(normalized)


def _require_unique_discretizations(
    values: Sequence[DiscretizationMetadata],
    owner: str,
) -> None:
    keys = [item.key for item in values]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{owner} contains duplicate discretizations")


def _require_nonempty(value: str, name: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} must be non-empty")


def _require_finite_complex(value: complex, name: str) -> None:
    z = complex(value)
    if not math.isfinite(z.real) or not math.isfinite(z.imag):
        raise ValueError(f"{name} must be finite")


def _require_finite_float(value: float | None, name: str) -> None:
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_float(value: float | None, name: str) -> None:
    _require_finite_float(value, name)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    _require_finite_complex(z, "serialized complex value")
    return [float(z.real), float(z.imag)]


def _complex_from_pair(value: Sequence[Any]) -> complex:
    if len(value) != 2:
        raise ValueError("serialized complex values must contain real and imaginary parts")
    result = complex(float(value[0]), float(value[1]))
    _require_finite_complex(result, "serialized complex value")
    return result


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _index_hash(indices: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


__all__ = [
    "CANONICAL_SELECTOR_STATES",
    "ContourLevelEvidence",
    "ContourProfile",
    "DeclaredPole",
    "DiscretizationMetadata",
    "FallbackProbeProfile",
    "FrequencyNormalization",
    "LocalRationalOperator",
    "LogDeterminant",
    "PredictionManifest",
    "ProbeEvidence",
    "ProbeTierProfile",
    "RootSelectionEvidence",
    "ScanScope",
    "SelectorFinalState",
    "SelectorProfile",
    "SelectorTiming",
    "SourceProfile",
    "SourceSpectrumProvenance",
    "SpectrumAccounting",
    "StructuralPoleEvidence",
    "TierSelectionResult",
    "ToleranceProfile",
    "validate_root_evidence_accounting",
]

from __future__ import annotations

"""Structural-pole matching backed by McMillan multiplicity certificates."""

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

from .hh_local_selector import HHRationalOperator
from .hh_minimal import HHMcMillanCertificate, build_hh_minimal_realization
from .multidiscretization_labels import PoleRemovalResult, remove_declared_poles


@dataclass(frozen=True)
class StructuralPoleProvenance:
    schema_version: int
    source_family: str
    certificate_kind: str
    grouping_mode: str
    original_size: int
    minimal_size: int
    structural_pole_count: int
    reconstructed_root_count: int
    count_closure: bool
    ambiguous_rank_count: int
    rank_tolerance: float
    structural_poles_sha256: str
    match_tolerance: float
    max_normalized_match_distance: float
    raw_csnj_used_as_deletion_list: bool = False
    certificate_scope: str = "full_mcmillan_minimal"
    certified_group_count: int = 0
    pole_group_count: int = 0
    tentative_structural_pole_count: int = 0
    retained_root_count: int = 0
    removal_count_closure: bool = True
    mcmillan_minimal: bool = True
    uncertified_pole_location_count: int = 0
    uncertified_poles_sha256: str = ""
    uncertified_pole_guard_tolerance: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CertifiedPoleRemoval:
    removal: PoleRemovalResult
    structural_poles_normalized: np.ndarray
    provenance: StructuralPoleProvenance
    uncertified_pole_locations_normalized: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.complex128)
    )


def uncertified_pole_proximity_mask(
    source_roots_normalized: np.ndarray,
    certified: CertifiedPoleRemoval,
) -> np.ndarray:
    """Flag roots that must remain unresolved near an uncertified pole group.

    This is a conservative information-boundary rule, not a pole deletion
    rule.  A flagged source root stays in the retained spectrum and cannot be
    promoted to a stable or rejected state until the pole multiplicity is
    independently certified.
    """

    roots = np.asarray(source_roots_normalized, dtype=np.complex128).reshape(-1)
    locations = np.asarray(
        certified.uncertified_pole_locations_normalized,
        dtype=np.complex128,
    ).reshape(-1)
    if locations.size == 0:
        result = np.zeros(roots.size, dtype=bool)
        result.setflags(write=False)
        return result
    distances = np.abs(locations[:, None] - roots[None, :]) / (
        1.0
        + np.maximum(
            np.abs(locations)[:, None],
            np.abs(roots)[None, :],
        )
    )
    result = np.min(distances, axis=0) <= float(
        certified.provenance.uncertified_pole_guard_tolerance
    )
    # A fully certified deletion wins only for its own assigned occurrence;
    # the two sets should normally be disjoint, but make that invariant
    # explicit for overlapping numerical neighborhoods.
    result[np.asarray(certified.removal.removed_indices, dtype=np.int64)] = False
    result.setflags(write=False)
    return result


def remove_hh_structural_poles(
    source_roots_normalized: np.ndarray,
    operator: HHRationalOperator,
    *,
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
    ambiguity_policy: Literal["conservative_retain", "abstain"] = (
        "conservative_retain"
    ),
    uncertified_pole_guard_tolerance: float = 3.0e-6,
) -> CertifiedPoleRemoval:
    """Remove only poles certified by the HH minimal realization.

    The raw coefficient array ``csnj`` remains available on the local
    rational operator for contour geometry.  It is deliberately not used as
    the source-spectrum deletion list because it does not encode the
    McMillan pole multiplicities.
    """

    if ambiguity_policy not in {"conservative_retain", "abstain"}:
        raise ValueError(
            "ambiguity_policy must be 'conservative_retain' or 'abstain'"
        )
    if uncertified_pole_guard_tolerance <= 0.0:
        raise ValueError("uncertified_pole_guard_tolerance must be positive")

    realization = build_hh_minimal_realization(operator.coefficients)
    certificate = realization.certificate
    _require_usable_certificate(
        certificate,
        allow_ambiguous=ambiguity_policy == "conservative_retain",
    )
    roots = np.asarray(source_roots_normalized, dtype=np.complex128).reshape(-1)
    if roots.size != certificate.original_size:
        raise ValueError(
            f"source root count {roots.size} does not match certificate size "
            f"{certificate.original_size}"
        )
    certified_chunks: list[np.ndarray] = []
    uncertified_locations: list[complex] = []
    certified_group_count = 0
    for group in realization.groups:
        if group.rank_ambiguous:
            # The tentative rank must not be converted into a deletion count.
            # Preserve every nearby source root for downstream abstention.
            uncertified_locations.append(complex(group.pole))
            continue
        certified_group_count += 1
        if group.known_pole_multiplicity:
            certified_chunks.append(
                np.full(
                    group.known_pole_multiplicity,
                    group.pole,
                    dtype=np.complex128,
                )
            )
    certified_poles = (
        np.concatenate(certified_chunks)
        if certified_chunks
        else np.empty(0, dtype=np.complex128)
    )
    poles = np.asarray(
        certified_poles / operator.normalization,
        dtype=np.complex128,
    )
    uncertified = np.asarray(
        uncertified_locations,
        dtype=np.complex128,
    ) / operator.normalization
    if not certificate.ambiguous_rank_count and poles.size != certificate.known_pole_count:
        raise AssertionError("HH structural-pole certificate size mismatch")

    last_error: ValueError | None = None
    matched: PoleRemovalResult | None = None
    selected_tolerance: float | None = None
    for tolerance in match_tolerances:
        if tolerance <= 0.0:
            raise ValueError("match tolerances must be positive")
        try:
            matched = remove_declared_poles(
                roots,
                poles,
                match_tolerance=float(tolerance),
            )
            selected_tolerance = float(tolerance)
            break
        except ValueError as exc:
            last_error = exc
    if matched is None or selected_tolerance is None:
        raise ValueError(f"could not match certified HH structural poles: {last_error}")
    if matched.removed_indices.size != poles.size:
        raise AssertionError("matched structural-pole multiplicity does not close")
    if (
        not certificate.ambiguous_rank_count
        and matched.retained_indices.size != certificate.minimal_size
    ):
        raise AssertionError("retained finite-root count does not equal minimal size")
    removal_count_closure = (
        matched.removed_indices.size + matched.retained_indices.size == roots.size
    )
    if not removal_count_closure:
        raise AssertionError("certified pole removal does not close the source root count")

    pole_bytes = np.ascontiguousarray(poles).view(np.uint8)
    uncertified_bytes = np.ascontiguousarray(uncertified).view(np.uint8)
    is_partial = bool(certificate.ambiguous_rank_count)
    provenance = StructuralPoleProvenance(
        schema_version=2,
        source_family="2025_bo_arbitrary_hh",
        certificate_kind=(
            "hh_mcmillan_groupwise_partial_v1"
            if is_partial
            else "hh_mcmillan_minimal_realization_v1"
        ),
        grouping_mode=certificate.grouping_mode,
        original_size=certificate.original_size,
        minimal_size=certificate.minimal_size,
        structural_pole_count=int(poles.size),
        reconstructed_root_count=certificate.reconstructed_root_count,
        count_closure=certificate.count_closure,
        ambiguous_rank_count=certificate.ambiguous_rank_count,
        rank_tolerance=certificate.rank_tolerance,
        structural_poles_sha256=hashlib.sha256(pole_bytes).hexdigest(),
        match_tolerance=selected_tolerance,
        max_normalized_match_distance=float(matched.max_normalized_match_distance),
        certificate_scope=(
            "unambiguous_groups_only" if is_partial else "full_mcmillan_minimal"
        ),
        certified_group_count=certified_group_count,
        pole_group_count=certificate.pole_group_count,
        tentative_structural_pole_count=certificate.known_pole_count,
        retained_root_count=int(matched.retained_indices.size),
        removal_count_closure=removal_count_closure,
        mcmillan_minimal=certificate.mcmillan_minimal,
        uncertified_pole_location_count=int(uncertified.size),
        uncertified_poles_sha256=hashlib.sha256(uncertified_bytes).hexdigest(),
        uncertified_pole_guard_tolerance=float(uncertified_pole_guard_tolerance),
    )
    poles.setflags(write=False)
    uncertified.setflags(write=False)
    return CertifiedPoleRemoval(
        removal=matched,
        structural_poles_normalized=poles,
        provenance=provenance,
        uncertified_pole_locations_normalized=uncertified,
    )


def _require_usable_certificate(
    certificate: HHMcMillanCertificate,
    *,
    allow_ambiguous: bool,
) -> None:
    if not certificate.count_closure:
        raise ValueError("HH McMillan certificate does not close the source root count")
    if certificate.ambiguous_rank_count and not allow_ambiguous:
        raise ValueError(
            "HH McMillan certificate contains rank-ambiguous pole groups; "
            "structural removal must abstain"
        )
    if not certificate.mcmillan_minimal and not (
        allow_ambiguous and certificate.ambiguous_rank_count
    ):
        raise ValueError("HH realization is not certified McMillan minimal")

from __future__ import annotations

"""Selector profiles for the 13 BO cases.

The HH v1 profiles were declared from the canonical case registry and solver
capabilities before reference evaluation. The legacy forward-J v2 policy was
repaired after the frozen Muschietti17 v1 output was compared with the archived
BO branch and independent discretization audit. The registry imports no
reference, branch-table, paper, or evaluation module; every prediction run is
reference-free, but the legacy v2 policy is development-informed rather than
a blind holdout protocol.

Probe rationale
---------------

* The firehose profile reproduces the supplied selector patch configuration
  exactly: an upper-J probe and one/two added harmonic shells are the primary
  near probes, the lower-J probe is the ambiguity fallback, and the core tier
  contains the supplied wider and crossed stresses.
* The other HH profiles apply the same one-coordinate-at-a-time construction.
  ICE starts at the smallest implemented J table, so its J probes are
  necessarily one-sided and explicitly recorded as such.
* Legacy ``j8_i10_opt3`` cases refine the J=8 source using J=12 and J=16.
  The coarser J=4 response is not an acceptance gate because a root can be
  converged as J increases while lying outside the J=4 local neighborhood.
  The historical beam-mode source uses ``j8_i8_opt2``, which exists only at
  J=8; any ``opt3`` probe is explicitly a *variant stress*, never described as
  a pure J perturbation.
* The unmagnetized ES1D case receives J-only probes because cyclotron-harmonic
  truncation is not an active approximation axis for that model.

All profiles use the frozen main-paper movement rule
``abs(delta_omega) / (1 + abs(omega)) <= 1e-3``.  Alternative tolerances belong
to post-freeze sensitivity analysis and are not selector defaults.
"""

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .case_registry import BOCaseRecord, get_case_record, list_case_records
from .selector_models import (
    ContourProfile,
    DiscretizationMetadata,
    FallbackProbeProfile,
    ProbeTierProfile,
    ScanScope,
    SelectorProfile,
    SourceProfile,
    ToleranceProfile,
)


HH_FAMILY = "2025_bo_arbitrary_hh"
LEGACY_FAMILY = "2018_bo_pdrk"

MAIN_NORMALIZED_MOVEMENT_EPSILON = 1.0e-3
MAIN_NORMALIZED_MOVEMENT_FORMULA = "abs(delta_omega)/(1+abs(omega))"

SUPPORTED_HH_J_POLE_ORDERS: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
SUPPORTED_LEGACY_J_POLE_ORDERS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "j8_i10_opt3": SUPPORTED_HH_J_POLE_ORDERS,
        "j8_i8_opt2": (8,),
    }
)

_CONTOUR = ContourProfile(
    sample_levels=(16, 32, 64, 128),
    consecutive_equal_counts=2,
    required_winding_number=1,
    rigorous=False,
)
_TOLERANCE = ToleranceProfile(
    normalized_movement_epsilon=MAIN_NORMALIZED_MOVEMENT_EPSILON
)


@dataclass(frozen=True)
class _ProbeSpec:
    pole_order: int
    harmonic_truncation: int
    variant: str


@dataclass(frozen=True)
class _ProbePolicy:
    near: tuple[_ProbeSpec, ...]
    fallback: _ProbeSpec
    core: tuple[_ProbeSpec, ...]
    j_rationale: str
    harmonic_rationale: str
    variant_rationale: str


def _spec(j_poles: int, n_harmonics: int, variant: str) -> _ProbeSpec:
    return _ProbeSpec(j_poles, n_harmonics, variant)


def _hh_policy(
    *,
    source_j: int,
    source_n: int,
    upper_near_j: int,
    fallback_j: int,
    upper_core_j: int,
    lower_core_j: int | None,
    ice_one_sided: bool = False,
) -> _ProbePolicy:
    variant = "finite_bo"
    near = (
        _spec(upper_near_j, source_n, variant),
        _spec(source_j, source_n + 1, variant),
        _spec(source_j, source_n + 2, variant),
    )
    core_items: list[_ProbeSpec] = []
    if lower_core_j is not None:
        core_items.append(_spec(lower_core_j, source_n, variant))
    core_items.extend(
        (
            _spec(upper_core_j, source_n, variant),
            _spec(source_j, source_n + 3, variant),
            _spec(source_j, source_n + 4, variant),
            *(
                _spec(upper_core_j, source_n + offset, variant)
                for offset in range(1, 5)
            ),
        )
    )
    if ice_one_sided:
        j_rationale = (
            "The source uses the smallest implemented J=4 table; J=8, J=12, "
            "and J=16 provide successively wider one-sided pole-order probes."
        )
    else:
        j_rationale = (
            "The fallback and primary probes bracket the source with the nearest "
            "implemented lower/upper J tables; the core probes widen that bracket."
        )
    return _ProbePolicy(
        near=near,
        fallback=_spec(fallback_j, source_n, variant),
        core=tuple(core_items),
        j_rationale=j_rationale,
        harmonic_rationale=(
            "Near probes add one and two symmetric cyclotron-harmonic shells; "
            "core probes add three/four shells and cross them with the wider J stress."
        ),
        variant_rationale="All HH probes use the implemented finite_bo coefficient family.",
    )


_HH_POLICIES: Mapping[str, _ProbePolicy] = MappingProxyType(
    {
        # This entry is intentionally explicit: changing it would invalidate
        # the supplied firehose scan-23 regression baseline.
        "bo2025_firehose_astfalk17_theta45": _hh_policy(
            source_j=24,
            source_n=2,
            upper_near_j=28,
            fallback_j=20,
            upper_core_j=32,
            lower_core_j=16,
        ),
        "bo2025_ringbeam_umeda12_theta40": _hh_policy(
            source_j=12,
            source_n=6,
            upper_near_j=16,
            fallback_j=8,
            upper_core_j=20,
            lower_core_j=4,
        ),
        "bo2025_shell_min15_theta89": _hh_policy(
            source_j=24,
            source_n=10,
            upper_near_j=28,
            fallback_j=20,
            upper_core_j=32,
            lower_core_j=16,
        ),
        "bo2025_ice_irvine18": _hh_policy(
            source_j=4,
            source_n=22,
            upper_near_j=8,
            fallback_j=12,
            upper_core_j=16,
            lower_core_j=None,
            ice_one_sided=True,
        ),
    }
)


def _legacy_standard_policy(source_n: int) -> _ProbePolicy:
    variant = "j8_i10_opt3"
    return _ProbePolicy(
        near=(
            _spec(12, source_n, variant),
            _spec(8, source_n + 1, variant),
            _spec(8, source_n + 2, variant),
        ),
        fallback=_spec(16, source_n, variant),
        core=(
            _spec(20, source_n, variant),
            _spec(8, source_n + 3, variant),
            _spec(8, source_n + 4, variant),
            *(_spec(20, source_n + offset, variant) for offset in range(1, 5)),
        ),
        j_rationale=(
            "J=12, J=16, and J=20 are successive higher-order refinements of the J=8 "
            "source. J=4 is excluded from the acceptance gate because lower-order "
            "drift cannot disprove forward convergence."
        ),
        harmonic_rationale=(
            "One/two added harmonic shells test adjacent truncations; three/four "
            "shells and crossed J=20 probes form the core stress tier."
        ),
        variant_rationale=(
            "Source and probes all use j8_i10_opt3; this implementation supports "
            "that coefficient family at every declared J table."
        ),
    )


def _legacy_es1d_policy(source_n: int) -> _ProbePolicy:
    variant = "j8_i10_opt3"
    return _ProbePolicy(
        near=(_spec(12, source_n, variant),),
        fallback=_spec(16, source_n, variant),
        core=(_spec(20, source_n, variant),),
        j_rationale=(
            "J=12 and J=16 refine the J=8 source in one direction; the lower J=4 "
            "response is excluded from the acceptance gate."
        ),
        harmonic_rationale=(
            "No harmonic probe is declared: this unmagnetized ES1D model does not "
            "use cyclotron-harmonic truncation as an active approximation axis."
        ),
        variant_rationale="All probes retain the source j8_i10_opt3 family.",
    )


def _legacy_beammode_policy(source_n: int) -> _ProbePolicy:
    source_variant = "j8_i8_opt2"
    stress_variant = "j8_i10_opt3"
    return _ProbePolicy(
        near=(
            _spec(8, source_n, stress_variant),
            _spec(8, source_n + 1, source_variant),
            _spec(8, source_n + 2, source_variant),
        ),
        fallback=_spec(16, source_n, stress_variant),
        core=(
            _spec(20, source_n, stress_variant),
            _spec(8, source_n + 3, source_variant),
            _spec(8, source_n + 4, source_variant),
            *(_spec(20, source_n + offset, stress_variant) for offset in range(1, 5)),
        ),
        j_rationale=(
            "The historical opt2 source has no non-J=8 table. J=16 is an upward "
            "combined J/variant stress in opt3, not a pure J perturbation."
        ),
        harmonic_rationale=(
            "Adjacent and wider harmonic-shell probes at J=8 retain opt2; crossed "
            "core probes use the separately declared opt3 stress family."
        ),
        variant_rationale=(
            "J=8 opt3 is an explicit fit-variant stress.  The profile never labels "
            "an opt2-to-opt3 comparison as a pole-order-only change."
        ),
    )


_LEGACY_POLICY_FACTORIES = MappingProxyType(
    {
        "bo2018_beammode": _legacy_beammode_policy,
        "bo2018_coldplasma": _legacy_standard_policy,
        "bo2018_darwin": _legacy_standard_policy,
        "bo2018_dispersionsurface": _legacy_standard_policy,
        "bo2018_es1d_beam": _legacy_es1d_policy,
        "bo2018_es3d_losscone": _legacy_standard_policy,
        "bo2018_loss_cone_mirror": _legacy_standard_policy,
        "bo2018_muschietti17": _legacy_standard_policy,
        "bo2018_ringbeam_umeda12": _legacy_standard_policy,
    }
)


def _operator_kind(record: BOCaseRecord) -> str:
    if record.source_family == HH_FAMILY:
        return "hh_finite_bo_3x3"
    if record.model.iem == 0:
        return "legacy_es_scalar"
    if record.model.iem == 2:
        return "legacy_darwin_3x3"
    return "legacy_em_3x3"


def _discretization(record: BOCaseRecord, spec: _ProbeSpec) -> DiscretizationMetadata:
    return DiscretizationMetadata(
        family=record.source_family,
        pole_order=spec.pole_order,
        harmonic_truncation=spec.harmonic_truncation,
        variant=spec.variant,
        parameters=(
            ("iem", record.model.iem),
            ("operator_kind", _operator_kind(record)),
        ),
    )


def _policy_for(record: BOCaseRecord) -> _ProbePolicy:
    if record.source_family == HH_FAMILY:
        return _HH_POLICIES[record.case_id]
    return _LEGACY_POLICY_FACTORIES[record.case_id](record.model.n_harmonics)


def _build_profile(record: BOCaseRecord) -> SelectorProfile:
    policy = _policy_for(record)
    source_variant = (
        "finite_bo" if record.source_family == HH_FAMILY else str(record.model.jpole_variant)
    )
    source_spec = _spec(
        record.model.j_poles,
        record.model.n_harmonics,
        source_variant,
    )
    return SelectorProfile(
        profile_id=(
            f"{record.case_id}__capability-predeclared-v1"
            if record.source_family == HH_FAMILY
            else f"{record.case_id}__forward-j-refinement-v2"
        ),
        family=record.source_family,
        source=SourceProfile(
            discretization=_discretization(record, source_spec),
            operator_id=_operator_kind(record),
        ),
        near=ProbeTierProfile(
            tuple(_discretization(record, item) for item in policy.near),
            require_all=True,
        ),
        fallback=FallbackProbeProfile(
            discretization=_discretization(record, policy.fallback),
            trigger_statuses=(
                "ambiguous",
                "newton_unresolved",
                "nonfinite_one_step",
            ),
        ),
        core=ProbeTierProfile(
            tuple(_discretization(record, item) for item in policy.core),
            require_all=True,
        ),
        contour=_CONTOUR,
        tolerance=_TOLERANCE,
        scan_scope=ScanScope(
            case_id=record.case_id,
            scan_indices=None,
            source_variant="registered_canonical",
            partition="full_registered_scan",
            held_out=False,
            expected_scan_count=record.scan.points,
        ),
        provenance=(
            "Predeclared from case_registry.ModelDescriptor and implemented J-pole/"
            "harmonic capabilities before any reference, branch, oracle, or label read."
            if record.source_family == HH_FAMILY else
            "Development-informed forward-J repair after post-freeze review of the "
            "Muschietti17 v1 archived BO branch and multi-discretization audit. "
            "The v2 prediction path itself does not read reference roots or labels."
        ),
        metadata={
            "declared_before_evaluation": record.source_family == HH_FAMILY,
            "evaluation_labels_read": record.source_family != HH_FAMILY,
            "prediction_labels_read": False,
            "harmonic_probe_rationale": policy.harmonic_rationale,
            "j_probe_rationale": policy.j_rationale,
            "movement_formula": MAIN_NORMALIZED_MOVEMENT_FORMULA,
            "profile_basis": (
                "case_registry_and_solver_capabilities_only"
                if record.source_family == HH_FAMILY else
                "case_registry_capabilities_and_muschietti17_v1_failure_analysis"
            ),
            "supported_j_tables": ",".join(
                str(value) for value in SUPPORTED_HH_J_POLE_ORDERS
            ),
            "variant_probe_rationale": policy.variant_rationale,
        },
    )


def _all_discretizations(profile: SelectorProfile) -> tuple[DiscretizationMetadata, ...]:
    fallback = () if profile.fallback is None else (profile.fallback.discretization,)
    core = () if profile.core is None else profile.core.discretizations
    return (
        profile.source.discretization,
        *profile.near.discretizations,
        *fallback,
        *core,
    )


def _validate_profile(record: BOCaseRecord, profile: SelectorProfile) -> None:
    source = profile.source.discretization
    if profile.family != record.source_family or source.family != record.source_family:
        raise AssertionError(f"profile family mismatch for {record.case_id}")
    if (
        source.pole_order != record.model.j_poles
        or source.harmonic_truncation != record.model.n_harmonics
    ):
        raise AssertionError(f"profile source discretization mismatch for {record.case_id}")
    if profile.scan_scope.case_id != record.case_id:
        raise AssertionError(f"profile scan scope mismatch for {record.case_id}")
    if profile.scan_scope.expected_scan_count != record.scan.points:
        raise AssertionError(f"profile scan count mismatch for {record.case_id}")
    if profile.scan_scope.scan_indices is not None:
        raise AssertionError(f"profile must cover the full registered scan: {record.case_id}")
    if profile.tolerance.normalized_movement_epsilon != MAIN_NORMALIZED_MOVEMENT_EPSILON:
        raise AssertionError(f"main movement tolerance changed for {record.case_id}")

    values = _all_discretizations(profile)
    if any(item.harmonic_truncation is None or item.harmonic_truncation <= 0 for item in values):
        raise AssertionError(f"invalid harmonic truncation in {record.case_id}")
    for item in values:
        if item.pole_order is None:
            raise AssertionError(f"missing J-pole order in {record.case_id}")
        if record.source_family == HH_FAMILY:
            if item.variant != "finite_bo" or item.pole_order not in SUPPORTED_HH_J_POLE_ORDERS:
                raise AssertionError(f"unsupported HH discretization in {record.case_id}")
        else:
            allowed = SUPPORTED_LEGACY_J_POLE_ORDERS.get(item.variant)
            if allowed is None or item.pole_order not in allowed:
                raise AssertionError(f"unsupported legacy discretization in {record.case_id}")

    probe_keys = tuple(item.key for item in values[1:])
    if len(probe_keys) != len(set(probe_keys)):
        raise AssertionError(f"duplicate probe discretization in {record.case_id}")
    if source.key in set(probe_keys):
        raise AssertionError(f"source discretization repeated as a probe in {record.case_id}")


def _build_registry() -> Mapping[str, SelectorProfile]:
    records = list_case_records()
    profiles = {record.case_id: _build_profile(record) for record in records}
    if len(profiles) != 13:
        raise AssertionError("the selector profile registry must contain exactly 13 cases")
    if set(_HH_POLICIES) != {
        record.case_id for record in records if record.source_family == HH_FAMILY
    }:
        raise AssertionError("HH profile policy keys differ from the canonical registry")
    if set(_LEGACY_POLICY_FACTORIES) != {
        record.case_id for record in records if record.source_family == LEGACY_FAMILY
    }:
        raise AssertionError("legacy profile policy keys differ from the canonical registry")
    for record in records:
        _validate_profile(record, profiles[record.case_id])
    return MappingProxyType(profiles)


SELECTOR_PROFILES: Mapping[str, SelectorProfile] = _build_registry()


def list_selector_profiles() -> tuple[SelectorProfile, ...]:
    """Return all profiles in stable case-id order."""

    return tuple(SELECTOR_PROFILES[case_id] for case_id in sorted(SELECTOR_PROFILES))


def get_selector_profile(case_id: str) -> SelectorProfile:
    """Return one immutable versioned profile by canonical case id."""

    try:
        return SELECTOR_PROFILES[case_id]
    except KeyError as exc:
        choices = ", ".join(sorted(SELECTOR_PROFILES))
        raise KeyError(f"Unknown selector profile {case_id!r}; choices: {choices}") from exc


def selector_profile_registry_sha256() -> str:
    """Hash the deterministic 13-profile JSON representation."""

    payload = {
        profile.scan_scope.case_id: profile.to_dict()
        for profile in list_selector_profiles()
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Import-time checks keep programmatic callers from silently drifting away from
# the canonical registry, while this explicit lookup proves every key resolves.
for _case_id in SELECTOR_PROFILES:
    get_case_record(_case_id)


__all__ = [
    "HH_FAMILY",
    "LEGACY_FAMILY",
    "MAIN_NORMALIZED_MOVEMENT_EPSILON",
    "MAIN_NORMALIZED_MOVEMENT_FORMULA",
    "SELECTOR_PROFILES",
    "SUPPORTED_HH_J_POLE_ORDERS",
    "SUPPORTED_LEGACY_J_POLE_ORDERS",
    "get_selector_profile",
    "list_selector_profiles",
    "selector_profile_registry_sha256",
]

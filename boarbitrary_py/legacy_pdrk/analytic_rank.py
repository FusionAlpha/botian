from __future__ import annotations

"""Analytic low-rank certificates for legacy magnetized EM residues.

The registered legacy EM coefficient formula has a rank-two structure for a
single magnetized, collisionless auxiliary state.  This module keeps that
special-case proof separate from the general SVD-based McMillan realization:
it supplies only a conservative rank upper bound and never changes the
numerical rank selected by :mod:`boarbitrary_py.legacy_pdrk.minimal`.
"""

from dataclasses import dataclass

import numpy as np

from .coefficients import effective_em_wavevector
from .config import LegacyCoefficients, LegacyContext, LegacyScanPoint


LEGACY_MAGNETIZED_SINGLETON_RANK2_DEFECT_TOLERANCE = (
    256.0 * np.finfo(float).eps
)


def _readonly(values: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LegacyMagnetizedRank2Certificate:
    """One audited ``B=L@R`` factorization for a magnetized 3x3 residue."""

    coefficient_index: int
    species_index: int
    harmonic: int
    pole_index: int
    pole: complex
    left_factor: np.ndarray
    right_factor: np.ndarray
    relative_factor_defect: float
    defect_tolerance: float
    certified: bool
    proof_scope: str

    @property
    def rank_upper_bound(self) -> int | None:
        return 2 if self.certified else None

    @property
    def structural_multiplicity_floor(self) -> int:
        return 1 if self.certified else 0


def certify_magnetized_singleton_rank2(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    coefficients: LegacyCoefficients,
    coefficient_index: int,
    *,
    defect_tolerance: float = LEGACY_MAGNETIZED_SINGLETON_RANK2_DEFECT_TOLERANCE,
) -> LegacyMagnetizedRank2Certificate | None:
    """Return a conservative rank-two certificate for one eligible residue.

    For the collisionless magnetized formula, let

    ``a = n*Omega/kx + vdx`` and ``h = cz*vtz + vdz``.

    With zero collision frequency, the three columns of every radial component
    have the form ``[a*q, y, h*q]``.  The sum over radial components preserves
    that relation, hence the 3x3 residue has rank at most two.  We construct
    the corresponding two-column factorization and require a strict normalized
    reconstruction check as a guard against formula or indexing drift.

    The function deliberately returns ``None`` outside this narrow proof
    scope.  Repeated-pole groups must call it only for singleton groups; sums
    of two rank-two residues can have rank three.
    """

    if defect_tolerance <= 0.0:
        raise ValueError("defect_tolerance must be positive")
    if ctx.case.model.iem not in {1, 2} or coefficients.em_residues is None:
        return None
    index = int(coefficient_index)
    if not 0 <= index < ctx.snj:
        raise IndexError(f"coefficient_index {index} is outside [0, {ctx.snj - 1}]")
    if bool(ctx.index_map.unmagnetized[index]):
        return None

    species = int(ctx.index_map.species[index])
    harmonic = int(ctx.index_map.harmonic_or_sigma[index])
    pole_index = int(ctx.index_map.pole[index])
    # The first production certificate intentionally matches the attachment's
    # collisionless derivation.  Other parameter regimes keep the
    # existing numerical SVD/abstention behavior.
    if float(ctx.nus[species]) != 0.0:
        return None

    residue = np.asarray(coefficients.em_residues[index], dtype=np.complex128)
    pole = complex(coefficients.poles[index])
    if residue.shape != (3, 3):
        raise ValueError("legacy EM residue must be 3x3")
    if not (
        np.isfinite(residue.real).all()
        and np.isfinite(residue.imag).all()
        and np.isfinite(pole.real)
        and np.isfinite(pole.imag)
    ):
        return None

    _, kx, kz = effective_em_wavevector(point)
    cz_signed = -ctx.czj if kz < 0.0 else ctx.czj
    a_value = harmonic * ctx.wcs[species] / kx + ctx.vdsx[species]
    h_value = (
        cz_signed[pole_index] * ctx.vtzs[species] + ctx.vdsz[species]
    )

    # Choose the better-scaled of the analytically proportional first/third
    # columns.  This is the same factorization in two coordinate charts and also
    # handles a=0 or h=0 without division by zero.
    if abs(a_value) >= abs(h_value) and a_value != 0.0:
        ratio = h_value / a_value
        left = np.column_stack((residue[:, 0], residue[:, 1]))
        right = np.asarray(
            ((1.0, 0.0, ratio), (0.0, 1.0, 0.0)),
            dtype=np.complex128,
        )
        proof_scope = "collisionless_column3_over_column1"
    else:
        ratio = 0.0 + 0.0j if h_value == 0.0 else a_value / h_value
        left = np.column_stack((residue[:, 1], residue[:, 2]))
        right = np.asarray(
            ((0.0, 1.0, 0.0), (ratio, 0.0, 1.0)),
            dtype=np.complex128,
        )
        proof_scope = "collisionless_column1_over_column3"

    scale = max(float(np.linalg.norm(residue, ord="fro")), np.finfo(float).tiny)
    defect = float(np.linalg.norm(residue - left @ right, ord="fro") / scale)
    certified = bool(np.isfinite(defect) and defect <= defect_tolerance)
    return LegacyMagnetizedRank2Certificate(
        coefficient_index=index,
        species_index=species,
        harmonic=harmonic,
        pole_index=pole_index,
        pole=pole,
        left_factor=_readonly(left, dtype=np.complex128),
        right_factor=_readonly(right, dtype=np.complex128),
        relative_factor_defect=defect,
        defect_tolerance=float(defect_tolerance),
        certified=certified,
        proof_scope=proof_scope,
    )

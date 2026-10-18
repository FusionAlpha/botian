from __future__ import annotations

"""Local rational operators for the legacy BO/PDRK linearizations.

The operators in this module are Schur complements of the already verified
legacy full pencils.  They accept normalized frequency ``omega / ctx.wn`` and
do not solve an alternative full spectrum.  The zero-frequency constraint is
deliberately excluded: both operators divide the field Schur complement by
``omega`` and therefore must only be evaluated away from zero.
"""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..selector_models import (
    DeclaredPole,
    DiscretizationMetadata,
    FrequencyNormalization,
    LogDeterminant,
)
from .coefficients import (
    effective_em_wavevector,
    evaluate_em3d_coefficients,
    evaluate_es3d_coefficients,
)
from .config import (
    LegacyCoefficients,
    LegacyContext,
    LegacyMcMillanCertificate,
    LegacyMinimalGroup,
    LegacyScanPoint,
)
from .matrix import assemble_em3d_matrix, assemble_es3d_matrix
from .minimal import build_minimal_realization


ComplexArray = NDArray[np.complex128]


def _readonly(values: np.ndarray, *, dtype=np.complex128) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _stable_slogdet(tensor: ComplexArray) -> LogDeterminant:
    try:
        sign, log_abs = np.linalg.slogdet(np.asarray(tensor, dtype=np.complex128))
    except np.linalg.LinAlgError:
        return LogDeterminant(log_abs_determinant=None, phase=None, finite=False)
    finite = bool(
        np.isfinite(sign.real)
        and np.isfinite(sign.imag)
        and np.isfinite(log_abs)
        and abs(sign) > np.finfo(float).tiny
    )
    if not finite:
        return LogDeterminant(log_abs_determinant=None, phase=None, finite=False)
    return LogDeterminant(
        log_abs_determinant=float(log_abs),
        phase=float(np.angle(sign)),
        finite=True,
    )


def _discretization(ctx: LegacyContext) -> DiscretizationMetadata:
    model = ctx.case.model
    operator_kind = "es_scalar" if model.iem == 0 else ("darwin_3x3" if model.iem == 2 else "em_3x3")
    return DiscretizationMetadata(
        family=ctx.case.source_family,
        pole_order=int(model.j_poles),
        harmonic_truncation=int(model.n_harmonics),
        variant=model.jpole_variant,
        parameters=(("iem", int(model.iem)), ("operator_kind", operator_kind)),
    )


def _normalization(ctx: LegacyContext) -> FrequencyNormalization:
    return FrequencyNormalization(
        scale=float(ctx.wn),
        reference="LegacyContext.wn",
        normalized_units="omega_over_wn",
    )


def _pole_certificate(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    coefficients: LegacyCoefficients,
    *,
    rank_tolerance: float,
) -> tuple[
    tuple[DeclaredPole, ...],
    tuple[LegacyMinimalGroup, ...],
    LegacyMcMillanCertificate,
    str,
]:
    """Return the certified pole multiplicities used by the selector.

    The residue-rank calculation that determines structural multiplicity is
    independent of the field mass block.  The existing minimal-realization
    builder does not solve the singular Darwin pencil, so for ``iem=2`` we run
    exactly that rank calculation through an otherwise identical ``iem=1``
    context and retain only its pole groups and McMillan accounting.
    """

    certificate_ctx = ctx
    provenance_suffix = "native minimal realization"
    if ctx.case.model.iem == 2:
        certificate_ctx = replace(
            ctx,
            case=replace(
                ctx.case,
                model=replace(ctx.case.model, iem=1),
            ),
        )
        provenance_suffix = "mass-independent EM residue-rank certificate for Darwin"

    realization = build_minimal_realization(
        certificate_ctx,
        point,
        coefficients=coefficients,
        rank_tolerance=rank_tolerance,
    )
    provenance = (
        "LegacyMinimalGroup.known_pole_multiplicity; exact-bitwise pole grouping; "
        f"{provenance_suffix}"
    )
    declared: list[DeclaredPole] = []
    for group_index, group in enumerate(realization.groups):
        if group.known_pole_multiplicity == 0:
            continue
        declared.append(
            DeclaredPole(
                value_normalized=group.pole / ctx.wn,
                multiplicity=int(group.known_pole_multiplicity),
                provenance=provenance,
                certificate_id=(
                    f"{ctx.case.case_id}:scan-{point.flat_index}:group-{group_index}:"
                    f"rank-tolerance-{rank_tolerance:.3e}"
                ),
            )
        )
    if sum(item.multiplicity for item in declared) != realization.certificate.known_pole_count:
        raise AssertionError("Declared legacy pole multiplicities do not close to the certificate")
    return (
        tuple(declared),
        realization.groups,
        realization.certificate,
        provenance,
    )


@dataclass(frozen=True)
class LegacyESRationalOperator:
    """Scalar electric-field Schur operator for a legacy ES pencil."""

    ctx: LegacyContext
    point: LegacyScanPoint
    coefficients: LegacyCoefficients
    normalization: FrequencyNormalization
    discretization: DiscretizationMetadata
    declared_poles: tuple[DeclaredPole, ...]
    structural_groups: tuple[LegacyMinimalGroup, ...]
    mcmillan_certificate: LegacyMcMillanCertificate
    pole_certificate_provenance: str
    constant_field_block: ComplexArray

    def tensor(self, root_normalized: complex) -> ComplexArray:
        z_normalized = complex(root_normalized)
        z = self.normalization.dimensionalize(z_normalized)
        if z == 0.0:
            raise ZeroDivisionError("The legacy ES field Schur operator excludes omega=0")
        residues = self.coefficients.es_residues
        if residues is None:
            raise ValueError("Legacy ES operator requires scalar residues")
        with np.errstate(all="ignore"):
            value = 1.0 + np.sum(residues / (z - self.coefficients.poles))
        return np.asarray([[value]], dtype=np.complex128)

    def tensor_derivative(self, root_normalized: complex) -> ComplexArray:
        z_normalized = complex(root_normalized)
        z = self.normalization.dimensionalize(z_normalized)
        if z == 0.0:
            raise ZeroDivisionError("The legacy ES field Schur derivative excludes omega=0")
        residues = self.coefficients.es_residues
        if residues is None:
            raise ValueError("Legacy ES operator requires scalar residues")
        with np.errstate(all="ignore"):
            value = -self.normalization.scale * np.sum(
                residues / (z - self.coefficients.poles) ** 2
            )
        return np.asarray([[value]], dtype=np.complex128)

    def slogdet(self, root_normalized: complex) -> LogDeterminant:
        return _stable_slogdet(self.tensor(root_normalized))

    @property
    def declared_poles_normalized(self) -> ComplexArray:
        return _expanded_declared_poles(self.declared_poles)


@dataclass(frozen=True)
class LegacyEMRationalOperator:
    """Three-field rational Schur operator for EM or generalized Darwin BO."""

    ctx: LegacyContext
    point: LegacyScanPoint
    coefficients: LegacyCoefficients
    normalization: FrequencyNormalization
    discretization: DiscretizationMetadata
    declared_poles: tuple[DeclaredPole, ...]
    structural_groups: tuple[LegacyMinimalGroup, ...]
    mcmillan_certificate: LegacyMcMillanCertificate
    pole_certificate_provenance: str
    electric_mass_block: ComplexArray
    constant_field_block: ComplexArray
    vacuum_field_block: ComplexArray

    def tensor(self, root_normalized: complex) -> ComplexArray:
        z_normalized = complex(root_normalized)
        z = self.normalization.dimensionalize(z_normalized)
        if z == 0.0:
            raise ZeroDivisionError("The legacy EM/Darwin field Schur operator excludes omega=0")
        residues = self.coefficients.em_residues
        if residues is None:
            raise ValueError("Legacy EM/Darwin operator requires 3x3 residues")
        with np.errstate(all="ignore"):
            dynamic = np.sum(
                residues / (z * (z - self.coefficients.poles))[:, None, None],
                axis=0,
            )
            return (
                self.electric_mass_block
                + (self.constant_field_block + self.vacuum_field_block) / z**2
                + dynamic
            )

    def tensor_derivative(self, root_normalized: complex) -> ComplexArray:
        z_normalized = complex(root_normalized)
        z = self.normalization.dimensionalize(z_normalized)
        if z == 0.0:
            raise ZeroDivisionError("The legacy EM/Darwin field Schur derivative excludes omega=0")
        residues = self.coefficients.em_residues
        if residues is None:
            raise ValueError("Legacy EM/Darwin operator requires 3x3 residues")
        with np.errstate(all="ignore"):
            factors = -(2.0 * z - self.coefficients.poles) / (
                z**2 * (z - self.coefficients.poles) ** 2
            )
            derivative_dimensional = (
                -2.0 * (self.constant_field_block + self.vacuum_field_block) / z**3
                + np.sum(residues * factors[:, None, None], axis=0)
            )
        return self.normalization.scale * derivative_dimensional

    def slogdet(self, root_normalized: complex) -> LogDeterminant:
        return _stable_slogdet(self.tensor(root_normalized))

    @property
    def declared_poles_normalized(self) -> ComplexArray:
        return _expanded_declared_poles(self.declared_poles)

    @property
    def is_darwin(self) -> bool:
        return self.ctx.case.model.iem == 2


LegacyRationalOperator = LegacyESRationalOperator | LegacyEMRationalOperator


def _expanded_declared_poles(declared: tuple[DeclaredPole, ...]) -> ComplexArray:
    if not declared:
        return _readonly(np.empty(0, dtype=np.complex128))
    values = np.concatenate(
        [
            np.full(item.multiplicity, item.value_normalized, dtype=np.complex128)
            for item in declared
        ]
    )
    return _readonly(values)


def _em_vacuum_field_block(ctx: LegacyContext, point: LegacyScanPoint) -> ComplexArray:
    k, kx, kz = effective_em_wavevector(point)
    c2 = ctx.constants.c2
    return _readonly(
        np.asarray(
            (
                (-c2 * kz**2, 0.0, c2 * kx * kz),
                (0.0, -c2 * k**2, 0.0),
                (c2 * kx * kz, 0.0, -c2 * kx**2),
            ),
            dtype=np.complex128,
        )
    )


def _em_electric_mass_block(ctx: LegacyContext, point: LegacyScanPoint) -> ComplexArray:
    if ctx.case.model.iem == 1:
        return _readonly(np.eye(3, dtype=np.complex128))
    if ctx.case.model.iem != 2:
        raise ValueError(f"EM/Darwin operator requires iem=1 or iem=2, got {ctx.case.model.iem}")
    k, kx, kz = effective_em_wavevector(point)
    # Preserve both the entries and floating-point accumulation order of
    # matrix._assemble_darwin_mass_matrix.  The order matters on the parallel
    # edge, where ``1 + O(1e-47) - 1`` is exactly zero in binary64.
    block = np.eye(3, dtype=np.complex128)
    for row, column, value in (
        (0, 0, kx**2 / k**2),
        (2, 0, kx * kz / k**2),
        (0, 2, kx**2 / k**2),
        (2, 2, kz**2 / k**2),
        (0, 0, -1.0),
        (1, 1, -1.0),
        (2, 2, -1.0),
    ):
        block[row, column] += value
    return _readonly(block)


def build_legacy_local_operator(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    coefficients: LegacyCoefficients | None = None,
    quadrature_order: int = 120,
    rank_tolerance: float = 1.0e-12,
) -> LegacyRationalOperator:
    """Build the family-correct local operator and structural-pole certificate."""

    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")
    normalization = _normalization(ctx)
    discretization = _discretization(ctx)

    if ctx.case.model.iem == 0:
        coeff = coefficients or evaluate_es3d_coefficients(
            ctx,
            point,
            quadrature_order=quadrature_order,
        )
        if coeff.es_residues is None or coeff.poles.shape != (ctx.snj,):
            raise ValueError("Legacy ES coefficient shapes do not match the context")
        declared, groups, certificate, provenance = _pole_certificate(
            ctx,
            point,
            coeff,
            rank_tolerance=rank_tolerance,
        )
        constant = _readonly(
            np.asarray([[-np.sum(coeff.es_residues)]], dtype=np.complex128)
        )
        return LegacyESRationalOperator(
            ctx=ctx,
            point=point,
            coefficients=coeff,
            normalization=normalization,
            discretization=discretization,
            declared_poles=declared,
            structural_groups=groups,
            mcmillan_certificate=certificate,
            pole_certificate_provenance=provenance,
            constant_field_block=constant,
        )

    if ctx.case.model.iem in {1, 2}:
        coeff = coefficients or evaluate_em3d_coefficients(
            ctx,
            point,
            quadrature_order=quadrature_order,
        )
        if (
            coeff.em_residues is None
            or coeff.em_constant is None
            or coeff.poles.shape != (ctx.snj,)
            or coeff.em_residues.shape != (ctx.snj, 3, 3)
            or coeff.em_constant.shape != (3, 3)
        ):
            raise ValueError("Legacy EM/Darwin coefficient shapes do not match the context")
        declared, groups, certificate, provenance = _pole_certificate(
            ctx,
            point,
            coeff,
            rank_tolerance=rank_tolerance,
        )
        return LegacyEMRationalOperator(
            ctx=ctx,
            point=point,
            coefficients=coeff,
            normalization=normalization,
            discretization=discretization,
            declared_poles=declared,
            structural_groups=groups,
            mcmillan_certificate=certificate,
            pole_certificate_provenance=provenance,
            electric_mass_block=_em_electric_mass_block(ctx, point),
            constant_field_block=_readonly(coeff.em_constant),
            vacuum_field_block=_em_vacuum_field_block(ctx, point),
        )

    raise ValueError(f"Unsupported legacy iem={ctx.case.model.iem}")


def evaluate_legacy_schur_complement_tensor(
    operator: LegacyRationalOperator,
    root_normalized: complex,
) -> ComplexArray:
    """Evaluate the same field Schur tensor directly from the full pencil.

    This diagnostic is intentionally separate from :meth:`tensor`: it assembles
    the full legacy matrix and performs a dense block solve, so it is suitable
    for parity tests rather than production root screening.
    """

    z = operator.normalization.dimensionalize(complex(root_normalized))
    if z == 0.0:
        raise ZeroDivisionError("The field Schur diagnostic excludes omega=0")

    ctx = operator.ctx
    if isinstance(operator, LegacyESRationalOperator):
        assembled = assemble_es3d_matrix(
            ctx,
            operator.point,
            coefficients=operator.coefficients,
        )
        retained = np.asarray([ctx.snj], dtype=np.int64)
    else:
        assembled = assemble_em3d_matrix(
            ctx,
            operator.point,
            coefficients=operator.coefficients,
        )
        retained = np.arange(ctx.snj3, ctx.snj3 + 3, dtype=np.int64)

    matrix = assembled.matrix.toarray()
    mass = (
        np.eye(ctx.nn, dtype=np.complex128)
        if assembled.mass_matrix is None
        else assembled.mass_matrix.toarray()
    )
    pencil = z * mass - matrix
    eliminated = np.setdiff1d(np.arange(ctx.nn, dtype=np.int64), retained)
    schur = pencil[np.ix_(retained, retained)] - pencil[np.ix_(retained, eliminated)] @ np.linalg.solve(
        pencil[np.ix_(eliminated, eliminated)],
        pencil[np.ix_(eliminated, retained)],
    )
    return np.asarray(schur / z, dtype=np.complex128)

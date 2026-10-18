from __future__ import annotations

"""Local, reduced-operator diagnostics for BO HH roots.

The routines in this module deliberately avoid a second large matrix
``eig`` call.  A candidate from one BO spectrum is checked with either

* the 3x3 rational Schur operator for another ``(J, N)`` pair, or
* the 3x3 exact-Z dispersion tensor already implemented in
  :mod:`boarbitrary_py.dispersion_hh`.

All public root arguments and return values are normalized by ``ctx.wn``.
The contour/Rouche checks are sampled numerical diagnostics, not interval-
verified proofs; the returned ``rigorous`` flag is therefore always False.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import ScanPoint, SolverContext
from .dispersion_hh import (
    HHRootTerm,
    evaluate_hh_root_determinant,
    precompute_hh_root_terms,
)
from .hh_cases import load_hh_case_inputs
from .hh_coefficients import HHMatrixCoefficients, compute_hh_coefficients
from .initialize import initialize_case
from .matrix_hh import assemble_hh_matrix_from_coefficients
from .scan import iter_scan_points


Array = np.ndarray


@dataclass(frozen=True)
class LocalTensorEvaluation:
    root_normalized: complex
    determinant: complex
    tensor_norm_2: float
    sigma_min: float
    normalized_sigma_min: float
    singular_values: Array
    right_mode: Array
    left_mode: Array

    def to_dict(self) -> dict[str, object]:
        return {
            "root_normalized": _complex_pair(self.root_normalized),
            "determinant": _complex_pair(self.determinant),
            "tensor_norm_2": self.tensor_norm_2,
            "sigma_min": self.sigma_min,
            "normalized_sigma_min": self.normalized_sigma_min,
            "singular_values": [float(value) for value in self.singular_values],
            "right_mode": [_complex_pair(value) for value in self.right_mode],
            "left_mode": [_complex_pair(value) for value in self.left_mode],
        }


@dataclass(frozen=True)
class LocalRootCorrection:
    initial_root: complex
    corrected_root: complex
    converged: bool
    iterations: int
    absolute_move: float
    normalized_sigma_min: float
    trust_radius: float
    message: str
    residual_history: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_root": _complex_pair(self.initial_root),
            "corrected_root": _complex_pair(self.corrected_root),
            "converged": self.converged,
            "iterations": self.iterations,
            "absolute_move": self.absolute_move,
            "normalized_sigma_min": self.normalized_sigma_min,
            "trust_radius": self.trust_radius,
            "message": self.message,
            "residual_history": [float(value) for value in self.residual_history],
        }


@dataclass(frozen=True)
class SampledContourDiagnostic:
    center: complex
    radius: float
    samples: int
    source_winding_number: int
    target_winding_number: int
    declared_pole_count: int
    max_rouche_ratio: float
    min_source_sigma: float
    status: str
    rigorous: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "center": _complex_pair(self.center),
            "radius": self.radius,
            "samples": self.samples,
            "source_winding_number": self.source_winding_number,
            "target_winding_number": self.target_winding_number,
            "declared_pole_count": self.declared_pole_count,
            "max_rouche_ratio": self.max_rouche_ratio,
            "min_source_sigma": self.min_source_sigma,
            "status": self.status,
            "rigorous": self.rigorous,
        }


class LocalHHOperator(Protocol):
    def tensor(self, root_normalized: complex) -> Array:
        ...

    @property
    def declared_poles_normalized(self) -> Array:
        ...


@dataclass(frozen=True)
class HHRationalOperator:
    """3x3 rational Schur operator induced by a finite BO linearization."""

    ctx: SolverContext
    point: ScanPoint
    coefficients: HHMatrixCoefficients
    case_id: str | None = None

    def tensor(self, root_normalized: complex) -> Array:
        z = complex(root_normalized)
        w = z * self.ctx.wn
        if w == 0.0:
            raise ZeroDivisionError("The reduced HH operator is singular at omega=0")

        coeff = self.coefficients
        with np.errstate(all="ignore"):
            dynamic = np.sum(
                coeff.bsnj / (w * (w - coeff.csnj))[:, None, None],
                axis=0,
            )
            static_and_vacuum = (_vacuum_matrix(self) + np.sum(coeff.bs, axis=0)) / w**2
            return np.eye(3, dtype=np.complex128) + static_and_vacuum + dynamic

    def tensor_derivative(self, root_normalized: complex) -> Array:
        """Return ``d tensor / d(omega/wn)`` using the finite-BO formula."""

        z = complex(root_normalized)
        w = z * self.ctx.wn
        if w == 0.0:
            raise ZeroDivisionError("The reduced HH operator derivative is singular at omega=0")
        coeff = self.coefficients
        static_and_vacuum = _vacuum_matrix(self) + np.sum(coeff.bs, axis=0)
        with np.errstate(all="ignore"):
            dynamic_factors = -(2.0 * w - coeff.csnj) / (w**2 * (w - coeff.csnj) ** 2)
            return self.ctx.wn * (
                -2.0 * static_and_vacuum / w**3
                + np.sum(coeff.bsnj * dynamic_factors[:, None, None], axis=0)
            )

    @property
    def declared_poles_normalized(self) -> Array:
        return np.asarray(self.coefficients.csnj / self.ctx.wn, dtype=np.complex128)

    @property
    def discretization(self) -> tuple[int, int]:
        return int(self.ctx.setup.j_poles), int(self.ctx.setup.n_harmonics)

    @property
    def normalization(self) -> float:
        return float(self.ctx.wn)


@dataclass(frozen=True)
class HHExactZOperator:
    """3x3 exact-Z HH dispersion tensor at a fixed harmonic truncation."""

    ctx: SolverContext
    point: ScanPoint
    terms: tuple[HHRootTerm, ...]

    def tensor(self, root_normalized: complex) -> Array:
        z = complex(root_normalized)
        if z == 0.0:
            raise ZeroDivisionError("The exact-Z HH tensor is singular at omega=0")
        return evaluate_hh_root_determinant(
            self.ctx,
            self.point,
            z * self.ctx.wn,
            self.terms,
        ).tensor

    @property
    def declared_poles_normalized(self) -> Array:
        return np.empty(0, dtype=np.complex128)


def build_hh_context(
    case_id: str,
    *,
    workspace: str | Path | None = None,
    cases_root: str | Path | None = None,
    capsule_dir: str | Path | None = None,
    j_poles: int,
    n_harmonics: int,
) -> SolverContext:
    """Load a registered HH capsule and override only its BO discretization.

    ``workspace`` is retained for compatibility with the supplied selector
    patch.  New callers may pass the current loader's ``cases_root`` or
    ``capsule_dir`` arguments directly.
    """

    if j_poles <= 0 or n_harmonics < 0:
        raise ValueError("j_poles must be positive and n_harmonics non-negative")
    resolved_cases_root = (
        Path(cases_root).resolve()
        if cases_root is not None
        else (Path(workspace).resolve() / "paper_cases" if workspace is not None else None)
    )
    inputs = load_hh_case_inputs(
        case_id,
        cases_root=resolved_cases_root,
        capsule_dir=capsule_dir,
    )
    setup = replace(
        inputs.setup,
        j_poles=int(j_poles),
        n_harmonics=int(n_harmonics),
    )
    return initialize_case(replace(inputs, setup=setup))


def build_rational_operator(
    case_id: str,
    *,
    workspace: str | Path | None = None,
    cases_root: str | Path | None = None,
    capsule_dir: str | Path | None = None,
    scan_index: int,
    j_poles: int,
    n_harmonics: int,
    quadrature_order: int = 80,
) -> HHRationalOperator:
    ctx = build_hh_context(
        case_id,
        workspace=workspace,
        cases_root=cases_root,
        capsule_dir=capsule_dir,
        j_poles=j_poles,
        n_harmonics=n_harmonics,
    )
    point = _scan_point(ctx, scan_index)
    coeff = compute_hh_coefficients(ctx, point, quadrature_order=quadrature_order)
    return HHRationalOperator(
        ctx=ctx,
        point=point,
        coefficients=coeff,
        case_id=case_id,
    )


def build_exact_z_operator(
    case_id: str,
    *,
    workspace: str | Path | None = None,
    cases_root: str | Path | None = None,
    capsule_dir: str | Path | None = None,
    scan_index: int,
    n_harmonics: int,
    quadrature_order: int = 80,
    quadrature_extent: float = 10.0,
) -> HHExactZOperator:
    # J does not enter the exact-Z tensor, but keeping the source case's J makes
    # the normalization and all other initialized quantities identical.
    resolved_cases_root = (
        Path(cases_root).resolve()
        if cases_root is not None
        else (Path(workspace).resolve() / "paper_cases" if workspace is not None else None)
    )
    inputs = load_hh_case_inputs(
        case_id,
        cases_root=resolved_cases_root,
        capsule_dir=capsule_dir,
    )
    ctx = initialize_case(
        replace(
            inputs,
            setup=replace(inputs.setup, n_harmonics=int(n_harmonics)),
        )
    )
    point = _scan_point(ctx, scan_index)
    terms = precompute_hh_root_terms(
        ctx,
        point,
        quadrature_order=quadrature_order,
        quadrature_extent=quadrature_extent,
    )
    return HHExactZOperator(ctx=ctx, point=point, terms=terms)


def evaluate_local_operator(
    operator: LocalHHOperator,
    root_normalized: complex,
) -> LocalTensorEvaluation:
    tensor = np.asarray(operator.tensor(root_normalized), dtype=np.complex128)
    left, singular_values, vh = np.linalg.svd(tensor)
    tensor_norm = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    return LocalTensorEvaluation(
        root_normalized=complex(root_normalized),
        determinant=complex(np.linalg.det(tensor)),
        tensor_norm_2=tensor_norm,
        sigma_min=sigma_min,
        normalized_sigma_min=float(sigma_min / (1.0 + tensor_norm)),
        singular_values=np.asarray(singular_values, dtype=float),
        right_mode=_canonicalize(vh.conj().T[:, -1]),
        left_mode=_canonicalize(left[:, -1]),
    )


def evaluate_schur_complement_tensor(
    operator: HHRationalOperator,
    root_normalized: complex,
) -> Array:
    """Eliminate the assembled finite-BO pencil and return its ``3x3`` tensor.

    This deliberately independent diagnostic starts from the current sparse
    full-matrix assembly.  It is intended for parity tests and spot audits,
    not for the production selector's hot path.
    """

    z = complex(root_normalized)
    if not np.isfinite(z.real) or not np.isfinite(z.imag) or z == 0.0:
        raise ValueError("root_normalized must be finite and nonzero")
    w = z * operator.ctx.wn
    matrix = assemble_hh_matrix_from_coefficients(operator.coefficients).toarray()
    pencil = matrix - w * np.eye(operator.coefficients.nn, dtype=np.complex128)
    field = np.arange(operator.coefficients.snj3, operator.coefficients.snj3 + 3)
    eliminated = np.concatenate(
        (
            np.arange(operator.coefficients.snj3),
            np.arange(operator.coefficients.snj3 + 3, operator.coefficients.nn),
        )
    )
    reduced = (
        pencil[np.ix_(field, field)]
        - pencil[np.ix_(field, eliminated)]
        @ np.linalg.solve(
            pencil[np.ix_(eliminated, eliminated)],
            pencil[np.ix_(eliminated, field)],
        )
    )
    return np.asarray(-reduced / w, dtype=np.complex128)


def correct_local_root(
    operator: LocalHHOperator,
    initial_root: complex,
    *,
    trust_radius: float,
    max_iterations: int = 16,
    residual_tolerance: float = 2.0e-10,
    derivative_step: float = 2.0e-6,
    min_step: float = 1.0e-13,
) -> LocalRootCorrection:
    """SVD-projected Newton correction constrained to one local root basin."""

    if trust_radius <= 0.0:
        raise ValueError("trust_radius must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    z0 = complex(initial_root)
    z = z0
    best_z = z
    best_residual = float("inf")
    history: list[float] = []
    message = "maximum iterations reached"

    for iteration in range(max_iterations):
        evaluation = evaluate_local_operator(operator, z)
        residual = evaluation.normalized_sigma_min
        history.append(residual)
        if residual < best_residual:
            best_residual = residual
            best_z = z
        if residual <= residual_tolerance:
            message = "residual tolerance reached"
            return LocalRootCorrection(
                initial_root=z0,
                corrected_root=z,
                converged=True,
                iterations=iteration,
                absolute_move=float(abs(z - z0)),
                normalized_sigma_min=residual,
                trust_radius=float(trust_radius),
                message=message,
                residual_history=tuple(history),
            )

        tensor = operator.tensor(z)
        left, _, vh = np.linalg.svd(tensor)
        u = left[:, -1]
        v = vh.conj().T[:, -1]
        h = derivative_step * (1.0 + abs(z))
        derivative = (operator.tensor(z + h) - operator.tensor(z - h)) / (2.0 * h)
        projected = np.vdot(u, tensor @ v)
        projected_derivative = np.vdot(u, derivative @ v)
        if not np.isfinite(projected_derivative) or abs(projected_derivative) < 1.0e-28:
            message = "projected derivative is singular"
            break

        step = -projected / projected_derivative
        if not np.isfinite(step):
            message = "non-finite Newton step"
            break

        remaining = trust_radius - abs(z - z0)
        if remaining <= min_step:
            message = "trust region exhausted"
            break
        if abs(step) > remaining:
            step *= remaining / abs(step)

        accepted = False
        candidate = z
        for _ in range(10):
            candidate = z + step
            candidate_residual = evaluate_local_operator(operator, candidate).normalized_sigma_min
            if np.isfinite(candidate_residual) and candidate_residual < residual:
                accepted = True
                break
            step *= 0.5
            if abs(step) <= min_step * (1.0 + abs(z)):
                break

        if not accepted:
            message = "backtracking could not reduce the residual"
            break
        z = candidate
        if abs(step) <= min_step * (1.0 + abs(z)):
            message = "Newton step stagnated"
            break

    final = evaluate_local_operator(operator, best_z)
    return LocalRootCorrection(
        initial_root=z0,
        corrected_root=best_z,
        converged=bool(final.normalized_sigma_min <= residual_tolerance),
        iterations=len(history),
        absolute_move=float(abs(best_z - z0)),
        normalized_sigma_min=final.normalized_sigma_min,
        trust_radius=float(trust_radius),
        message=message,
        residual_history=tuple(history),
    )


def local_isolation_radius(
    root: complex,
    spectrum: Array,
    declared_poles: Array,
    *,
    fraction: float = 0.25,
    maximum: float = 0.08,
    minimum: float = 1.0e-8,
) -> float:
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must lie in (0, 0.5)")
    z = complex(root)
    values = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
    finite = values[np.isfinite(values.real) & np.isfinite(values.imag)]
    distances = np.abs(finite - z)
    nonzero = distances[distances > 1.0e-13 * (1.0 + abs(z))]
    neighbor = float(np.min(nonzero)) if nonzero.size else float("inf")

    poles = np.asarray(declared_poles, dtype=np.complex128).reshape(-1)
    poles = poles[np.isfinite(poles.real) & np.isfinite(poles.imag)]
    pole_distance = float(np.min(np.abs(poles - z))) if poles.size else float("inf")
    scale = min(neighbor, pole_distance, maximum / fraction)
    radius = fraction * scale
    return float(max(minimum, min(maximum, radius)))


def nearest_declared_pole_distance(
    root: complex,
    declared_poles: Array,
) -> float:
    poles = np.asarray(declared_poles, dtype=np.complex128).reshape(-1)
    poles = poles[np.isfinite(poles.real) & np.isfinite(poles.imag)]
    if poles.size == 0:
        return float("inf")
    return float(np.min(np.abs(poles - complex(root))))


def winding_number(
    operator: LocalHHOperator,
    center: complex,
    radius: float,
    *,
    samples: int = 64,
) -> int:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if samples < 8:
        raise ValueError("samples must be at least 8")
    theta = 2.0 * np.pi * np.arange(samples, dtype=float) / samples
    phases = np.empty(samples, dtype=np.complex128)
    for index, angle in enumerate(theta):
        z = center + radius * np.exp(1j * angle)
        sign, log_abs = np.linalg.slogdet(
            np.asarray(operator.tensor(z), dtype=np.complex128)
        )
        if (
            not np.isfinite(sign.real)
            or not np.isfinite(sign.imag)
            or not np.isfinite(log_abs)
            or abs(sign) <= np.finfo(float).tiny
        ):
            raise FloatingPointError("non-finite or singular determinant on contour")
        phases[index] = sign / abs(sign)
    increments = np.angle(np.roll(phases, -1) * np.conjugate(phases))
    phase = float(np.sum(increments))
    return int(np.rint(phase / (2.0 * np.pi)))


def sampled_rouche_diagnostic(
    source: LocalHHOperator,
    target: LocalHHOperator,
    *,
    center: complex,
    radius: float,
    samples: int = 64,
) -> SampledContourDiagnostic:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    theta = 2.0 * np.pi * np.arange(samples, dtype=float) / samples
    ratios: list[float] = []
    source_determinants: list[complex] = []
    target_determinants: list[complex] = []
    source_sigmas: list[float] = []

    for angle in theta:
        z = center + radius * np.exp(1j * angle)
        source_tensor = source.tensor(z)
        target_tensor = target.tensor(z)
        singular_values = np.linalg.svd(source_tensor, compute_uv=False)
        sigma = float(singular_values[-1])
        source_sigmas.append(sigma)
        ratios.append(
            float(np.linalg.norm(target_tensor - source_tensor, ord=2) / max(sigma, np.finfo(float).tiny))
        )
        source_determinants.append(complex(np.linalg.det(source_tensor)))
        target_determinants.append(complex(np.linalg.det(target_tensor)))

    source_winding = _winding_from_values(np.asarray(source_determinants))
    target_winding = _winding_from_values(np.asarray(target_determinants))
    poles = np.asarray(source.declared_poles_normalized, dtype=np.complex128).reshape(-1)
    declared_pole_count = int(np.count_nonzero(np.abs(poles - center) < radius))
    max_ratio = float(np.max(ratios))
    min_sigma = float(np.min(source_sigmas))

    if declared_pole_count:
        status = "contains_declared_source_pole"
    elif max_ratio < 1.0 and source_winding == target_winding:
        status = "sampled_rouche_pass"
    elif target_winding == 0:
        status = "target_zero_count_is_zero"
    elif source_winding != target_winding:
        status = "winding_number_changed"
    else:
        status = "sampled_rouche_inconclusive"

    return SampledContourDiagnostic(
        center=complex(center),
        radius=float(radius),
        samples=int(samples),
        source_winding_number=source_winding,
        target_winding_number=target_winding,
        declared_pole_count=declared_pole_count,
        max_rouche_ratio=max_ratio,
        min_source_sigma=min_sigma,
        status=status,
        rigorous=False,
    )


def _scan_point(ctx: SolverContext, scan_index: int) -> ScanPoint:
    return next(iter_scan_points(ctx, scan_indices=[int(scan_index)]))


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


def _canonicalize(vector: Array) -> Array:
    result = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(result))
    if norm == 0.0 or not np.isfinite(norm):
        return result
    result = result / norm
    pivot = int(np.argmax(np.abs(result)))
    if abs(result[pivot]) > 0.0:
        result = result * np.exp(-1j * np.angle(result[pivot]))
    return result


def _winding_from_values(values: Array) -> int:
    ratios = np.roll(values, -1) / values
    return int(np.rint(float(np.sum(np.angle(ratios))) / (2.0 * np.pi)))


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]

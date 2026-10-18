from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


ROOT_METHOD_PURE_GROWTH = "pure_growth"
ROOT_METHOD_MATLAB_COMPLEX_LM = "matlab_complex_lm"
ROOT_BRANCH_INTERIOR = "interior_root_found"
ROOT_BRANCH_BOUNDARY = "boundary_floor"
ROOT_BRANCH_UNCONFIRMED = "unconfirmed"
ROOT_BRANCH_COMPLEX_LM = "complex_lm_root"


@dataclass(frozen=True)
class PureGrowthSearchConfig:
    y_min: float = 1.0e-4
    y_max: float = 0.12
    grid_points: int = 61
    boundary_atol: float = 1.0e-7
    success_residual: float = 1.0e-5
    xatol: float = 1.0e-10
    maxiter: int = 80


@dataclass(frozen=True)
class PureGrowthBranchResult:
    normalized_root: np.ndarray
    function_evaluations: int
    numerical_success: bool
    physical_branch_confirmed: bool
    status: int
    branch_status: str
    message: str


@dataclass(frozen=True)
class ComplexRootSearchConfig:
    xtol: float = 1.0e-12
    ftol: float = 1.0e-12
    gtol: float = 1.0e-12
    max_nfev: int = 1000
    success_residual: float = 1.0e-8
    method: str = "lm"


@dataclass(frozen=True)
class ComplexRootResult:
    normalized_root: np.ndarray
    function_evaluations: int
    numerical_success: bool
    status: int
    branch_status: str
    message: str
    residual_norm: float


DEFAULT_PURE_GROWTH_SEARCH = PureGrowthSearchConfig()
DEFAULT_COMPLEX_ROOT_SEARCH = ComplexRootSearchConfig()


def solve_pure_growth_branch(
    residual_norm_squared: Callable[[np.ndarray], float],
    *,
    normalized_guess: np.ndarray,
    xtol: float,
    config: PureGrowthSearchConfig = DEFAULT_PURE_GROWTH_SEARCH,
) -> PureGrowthBranchResult:
    """Search a pure-growth branch over imag(w)/wn and reject boundary floors."""

    def scalar_objective(y_value: float) -> float:
        return residual_norm_squared(np.array([0.0, y_value], dtype=float))

    grid = np.linspace(config.y_min, config.y_max, config.grid_points)
    values = np.array([scalar_objective(float(y_value)) for y_value in grid], dtype=float)
    nfev = int(grid.size)
    interior_minima = [
        index
        for index in range(1, grid.size - 1)
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]
    ]
    minima = interior_minima or [int(np.nanargmin(values))]

    candidates: list[tuple[float, np.ndarray, str, int, bool]] = []
    xatol = min(float(xtol), config.xatol)
    for index in minima:
        lower = float(grid[max(0, index - 1)])
        upper = float(grid[min(grid.size - 1, index + 1)])
        scalar_result = minimize_scalar(
            scalar_objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": xatol, "maxiter": config.maxiter},
        )
        nfev += int(scalar_result.nfev)
        candidate = np.array([0.0, float(scalar_result.x)], dtype=float)
        is_global_boundary = (
            np.isclose(candidate[1], config.y_min, atol=config.boundary_atol, rtol=0.0)
            or np.isclose(candidate[1], config.y_max, atol=config.boundary_atol, rtol=0.0)
        )
        branch_status = ROOT_BRANCH_BOUNDARY if is_global_boundary else ROOT_BRANCH_INTERIOR
        status = 10 if scalar_result.success and not is_global_boundary else -20
        candidates.append((float(scalar_result.fun), candidate, branch_status, status, bool(scalar_result.success)))

    candidates.sort(key=lambda item: item[0])
    best_objective, best_x, branch_status, status, optimizer_success = candidates[0]
    residual_norm = float(np.sqrt(best_objective))
    numerical_success = bool(residual_norm <= config.success_residual)
    physical_branch_confirmed = bool(
        optimizer_success and numerical_success and branch_status == ROOT_BRANCH_INTERIOR
    )
    message = (
        f"selected {branch_status} pure-growth candidate with "
        f"objective={best_objective:.3e}, residual_norm={residual_norm:.3e}"
    )
    return PureGrowthBranchResult(
        normalized_root=best_x,
        function_evaluations=nfev,
        numerical_success=numerical_success,
        physical_branch_confirmed=physical_branch_confirmed,
        status=status,
        branch_status=branch_status,
        message=message,
    )


def solve_complex_lm_root(
    residual_vector: Callable[[np.ndarray], np.ndarray],
    *,
    normalized_guess: np.ndarray,
    config: ComplexRootSearchConfig = DEFAULT_COMPLEX_ROOT_SEARCH,
) -> ComplexRootResult:
    """Solve a two-component complex root objective in normalized coordinates."""

    result = least_squares(
        residual_vector,
        np.asarray(normalized_guess, dtype=float),
        method=config.method,
        xtol=config.xtol,
        ftol=config.ftol,
        gtol=config.gtol,
        max_nfev=config.max_nfev,
    )
    residual = np.asarray(residual_vector(np.asarray(result.x, dtype=float)), dtype=float)
    residual_norm = float(np.linalg.norm(residual))
    numerical_success = bool(result.success and residual_norm <= config.success_residual)
    return ComplexRootResult(
        normalized_root=np.asarray(result.x, dtype=float),
        function_evaluations=int(result.nfev),
        numerical_success=numerical_success,
        status=int(result.status),
        branch_status=ROOT_BRANCH_COMPLEX_LM,
        message=f"complex LM result: {result.message}; residual_norm={residual_norm:.3e}",
        residual_norm=residual_norm,
    )

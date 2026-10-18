"""Pure-Python compatibility layer for BO/PDRK 2018-2019 cases."""

from .analytic_rank import (
    LEGACY_MAGNETIZED_SINGLETON_RANK2_DEFECT_TOLERANCE,
    LegacyMagnetizedRank2Certificate,
    certify_magnetized_singleton_rank2,
)
from .coefficients import evaluate_em3d_coefficients, evaluate_es3d_coefficients
from .initialize import initialize_legacy_case
from .io import LEGACY_CASE_IDS, load_legacy_case
from .local_operator import (
    LegacyEMRationalOperator,
    LegacyESRationalOperator,
    LegacyRationalOperator,
    build_legacy_local_operator,
    evaluate_legacy_schur_complement_tensor,
)
from .matrix import assemble_em3d_matrix, assemble_es3d_matrix
from .minimal import build_minimal_realization, solve_minimal_point
from .scan import iter_legacy_scan_points
from .solver import solve_legacy_point, solve_legacy_scan

__all__ = [
    "LEGACY_CASE_IDS",
    "LEGACY_MAGNETIZED_SINGLETON_RANK2_DEFECT_TOLERANCE",
    "LegacyEMRationalOperator",
    "LegacyESRationalOperator",
    "LegacyMagnetizedRank2Certificate",
    "LegacyRationalOperator",
    "assemble_em3d_matrix",
    "assemble_es3d_matrix",
    "build_legacy_local_operator",
    "build_minimal_realization",
    "certify_magnetized_singleton_rank2",
    "evaluate_em3d_coefficients",
    "evaluate_es3d_coefficients",
    "evaluate_legacy_schur_complement_tensor",
    "initialize_legacy_case",
    "iter_legacy_scan_points",
    "load_legacy_case",
    "solve_legacy_point",
    "solve_legacy_scan",
    "solve_minimal_point",
]

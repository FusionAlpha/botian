from __future__ import annotations

import os

import numpy as np
from scipy.linalg import eigvals
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional runtime dependency guard
    threadpool_limits = None

from .config import SolverContext


def _dense_eig_thread_limit() -> int | None:
    """Return the BLAS thread limit for medium-size dense eigensolves.

    For the default 732x732 BO HH matrix, many Windows OpenBLAS builds are
    faster with a small thread count than with all logical cores. Override:

        BO_DENSE_EIG_THREADS=1  # default
        BO_DENSE_EIG_THREADS=4  # try on high-clock workstations
        BO_DENSE_EIG_THREADS=0  # do not limit BLAS here
    """
    raw = os.environ.get("BO_DENSE_EIG_THREADS", "1").strip()
    if raw in {"", "0", "none", "None", "NONE"}:
        return None
    value = int(raw)
    if value < 1:
        return None
    return value


def solve_eigenvalues(matrix: csr_matrix, ctx: SolverContext, *, sigma: complex | None = None) -> np.ndarray:
    if ctx.setup.sp == 0:
        dense = matrix.toarray(order="F")
        if dense.dtype != np.complex128 or not dense.flags.f_contiguous:
            dense = np.asarray(dense, dtype=np.complex128, order="F")

        limit = _dense_eig_thread_limit()
        if threadpool_limits is not None and limit is not None:
            with threadpool_limits(limits=limit, user_api="blas"):
                return eigvals(dense, check_finite=False, overwrite_a=True)
        return eigvals(dense, check_finite=False, overwrite_a=True)

    if ctx.setup.sp in (1, 2):
        if sigma is None:
            sigma = ctx.setup.wg0 * ctx.wn
        values = eigs(matrix, k=ctx.nw, sigma=sigma, return_eigenvectors=False)
        return np.asarray(values)
    raise NotImplementedError(f"sp={ctx.setup.sp} is outside the HH matrix solver milestone.")


def sort_by_growth(values: np.ndarray) -> np.ndarray:
    return values[np.argsort(values.imag)[::-1]]

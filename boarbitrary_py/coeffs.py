from __future__ import annotations

from functools import lru_cache
from math import gamma, sqrt

import numpy as np
from scipy.special import jv, roots_legendre


def fun_in(n: int) -> float:
    if n % 2 != 0:
        return 0.0
    return gamma(0.5 * (n + 1)) / sqrt(np.pi)


@lru_cache(maxsize=8192)
def fun_an(n: int, a: float, d: float, m: int, p: int, order: int = 120, extent: float = 5.0) -> float:
    if m < 0:
        return 0.0
    y, weights = _quadrature_nodes(d, order, extent)
    values = jv(n, a * y) ** 2 * np.exp(-((y - d) ** 2)) * (y - d) ** m * y**p
    return float(np.sum(weights * values))


@lru_cache(maxsize=8192)
def fun_bn(n: int, a: float, d: float, m: int, p: int, order: int = 120, extent: float = 5.0) -> float:
    if m < 0:
        return 0.0
    y, weights = _quadrature_nodes(d, order, extent)
    derivative = 0.5 * (jv(n - 1, a * y) - jv(n + 1, a * y))
    values = jv(n, a * y) * derivative * np.exp(-((y - d) ** 2)) * (y - d) ** m * y**p
    return float(np.sum(weights * values))


@lru_cache(maxsize=8192)
def fun_cn(n: int, a: float, d: float, m: int, p: int, order: int = 120, extent: float = 5.0) -> float:
    if m < 0:
        return 0.0
    y, weights = _quadrature_nodes(d, order, extent)
    derivative = jv(n - 1, a * y) - jv(n + 1, a * y)
    values = 0.25 * derivative**2 * np.exp(-((y - d) ** 2)) * (y - d) ** m * y**p
    return float(np.sum(weights * values))


def precompute_abc(
    n: int,
    a: float,
    d: float,
    msmax: int,
    normalization: float,
    *,
    order: int = 120,
    extent: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ans = np.zeros((msmax + 4, 2), dtype=float)
    bns = np.zeros_like(ans)
    cns = np.zeros_like(ans)
    for m in range(msmax + 3):
        idx = m + 1
        ans[idx, 0] = 2.0 / normalization * fun_an(n, a, d, m, 0, order, extent)
        ans[idx, 1] = 2.0 / normalization * fun_an(n, a, d, m, 1, order, extent)
        bns[idx, 0] = 2.0 / normalization * fun_bn(n, a, d, m, 1, order, extent)
        bns[idx, 1] = 2.0 / normalization * fun_bn(n, a, d, m, 2, order, extent)
        cns[idx, 0] = 2.0 / normalization * fun_cn(n, a, d, m, 2, order, extent)
        cns[idx, 1] = 2.0 / normalization * fun_cn(n, a, d, m, 3, order, extent)
    return ans, bns, cns


def fun_gnmp(
    a: float,
    n: int,
    m: int,
    p: int,
    icase: int,
    order: int = 1024,
    extent: float = 500.0,
) -> complex:
    """Evaluate MATLAB funGnmp for scalar inputs using cached vectorized moments."""

    if icase not in {1, 2, 3}:
        raise ValueError(f"Unsupported fun_gnmp icase: {icase}")
    if p < 0:
        raise ValueError(f"Unsupported fun_gnmp p: {p}")
    m_abs = abs(int(m))
    values = _gnmp_positive_m_values(float(a), abs(int(n)), m_abs, int(order), float(extent))
    lookup = {
        (0, 1): values[0],
        (1, 1): values[1],
        (1, 2): values[2],
        (2, 2): values[3],
        (2, 3): values[4],
        (3, 3): values[5],
    }
    try:
        value = complex(lookup[(int(p), int(icase))][m_abs])
    except KeyError as exc:
        raise ValueError(f"Unsupported fun_gnmp p/icase pair: p={p}, icase={icase}") from exc
    return value if m >= 0 else value.conjugate()


def precompute_gnmp(
    n: int,
    a: float,
    msmax: int,
    *,
    order: int = 1024,
    extent: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute GG radial moments for m=-(msmax+1)..(msmax+1)."""

    mmax = int(msmax)
    mmin = -mmax
    values = _gnmp_positive_m_values(float(a), abs(int(n)), mmax + 1, int(order), float(extent))
    ans = np.zeros((2 * mmax + 3, 2), dtype=np.complex128)
    bns = np.zeros_like(ans)
    cns = np.zeros_like(ans)
    for m in range(-(mmax + 1), mmax + 2):
        idx = m + 1 - mmin
        src = abs(m)
        sign = 1 if m >= 0 else -1
        ans[idx, 0] = values[0][src] if sign > 0 else np.conj(values[0][src])
        ans[idx, 1] = values[1][src] if sign > 0 else np.conj(values[1][src])
        bns[idx, 0] = values[2][src] if sign > 0 else np.conj(values[2][src])
        bns[idx, 1] = values[3][src] if sign > 0 else np.conj(values[3][src])
        cns[idx, 0] = values[4][src] if sign > 0 else np.conj(values[4][src])
        cns[idx, 1] = values[5][src] if sign > 0 else np.conj(values[5][src])
    return ans, bns, cns


@lru_cache(maxsize=64)
def _legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    return roots_legendre(order)


def _quadrature_nodes(d: float, order: int, extent: float) -> tuple[np.ndarray, np.ndarray]:
    ymin = max(0.0, d - extent)
    ymax = extent + d
    x, w = _legendre(order)
    scale = 0.5 * (ymax - ymin)
    shift = 0.5 * (ymin + ymax)
    return scale * x + shift, scale * w


@lru_cache(maxsize=256)
def _gnmp_positive_m_values(
    a: float,
    n_abs: int,
    m_high: int,
    order: int,
    extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, weights = _gnmp_nodes(order, extent)
    m_values = np.arange(m_high + 1, dtype=int)
    ratio = (1.0 + 1j * x) / (1.0 - 1j * x)
    powers = ratio[None, :] ** m_values[:, None]
    jn = jv(n_abs, a * x)
    jprime = 0.5 * (jv(n_abs - 1, a * x) - jv(n_abs + 1, a * x))
    common = weights / (1.0 + x * x) ** 2

    def integrate(power: int, base: np.ndarray) -> np.ndarray:
        return powers @ (common * x**power * base)

    return (
        integrate(0, jn * jn),
        integrate(1, jn * jn),
        integrate(1, jn * jprime),
        integrate(2, jn * jprime),
        integrate(2, jprime * jprime),
        integrate(3, jprime * jprime),
    )


@lru_cache(maxsize=16)
def _gnmp_nodes(order: int, extent: float) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(order)
    scale = 0.5 * extent
    return scale * (nodes + 1.0), scale * weights

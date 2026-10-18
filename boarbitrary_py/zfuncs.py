from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import wofz

from .coeffs import fun_in


@lru_cache(maxsize=8192)
def fun_z_l(z: complex, ell: int) -> complex:
    """Hermite moment of the plasma dispersion function.

    This evaluates the Landau-contour moment

        Z_l(z) = int v^l exp(-v^2) / sqrt(pi) / (v - z) dv

    using the standard Faddeeva form for Z_0 and polynomial division for
    higher moments. It is the HH root counterpart of MATLAB `funZl`.
    """
    if ell < 0:
        return 0j
    z = complex(z)
    if _use_large_z_asymptotic(z, ell):
        return _fun_z_l_large_z(z, ell)

    z0 = 1j * np.sqrt(np.pi) * wofz(z)
    value = complex(z0)
    for order in range(1, ell + 1):
        value = z * value + fun_in(order - 1)
    return complex(value)


def _use_large_z_asymptotic(z: complex, ell: int) -> bool:
    magnitude_squared = abs(z) ** 2
    return bool(abs(z) >= 8.0 and magnitude_squared >= 4.0 * (ell + 1))


def _fun_z_l_large_z(z: complex, ell: int) -> complex:
    inverse_z = 1.0 / z
    inverse_power = inverse_z
    total = 0j
    previous_nonzero_term = float("inf")
    nonzero_terms = 0
    for offset in range(256):
        moment = fun_in(ell + offset)
        if moment != 0.0:
            term = -moment * inverse_power
            term_magnitude = abs(term)
            if term_magnitude > previous_nonzero_term:
                break
            total += term
            previous_nonzero_term = term_magnitude
            nonzero_terms += 1
            if nonzero_terms >= 3 and term_magnitude <= 4.0 * np.finfo(float).eps * max(abs(total), 1.0e-300):
                break
        inverse_power *= inverse_z

    residue_multiplier = 0.0
    if z.imag < 0.0:
        residue_multiplier = 2.0
    elif z.imag == 0.0:
        residue_multiplier = 1.0
    if residue_multiplier:
        total += residue_multiplier * 1j * np.sqrt(np.pi) * z**ell * np.exp(-(z**2))
    return complex(total)


def fun_z_l_array(z: complex, max_ell: int) -> np.ndarray:
    values = np.zeros(max_ell + 1, dtype=np.complex128)
    for ell in range(max_ell + 1):
        values[ell] = fun_z_l(z, ell)
    return values


@lru_cache(maxsize=8192)
def fun_z_lp(z: complex, ell: int, p: int) -> complex:
    """GPDF moment used by the GH/GG root formulations.

    This follows MATLAB `funZlp.m`. Negative `ell` values are intentionally
    zero for `p=0`; the `p=1` and `p=2` recurrences then define the values
    used by the GH/GG shifted-index sums.
    """
    z = complex(z)
    if p == 0:
        if ell == 0:
            return complex(-1j / (1 - 1j * z))
        if ell > 0:
            ratio = (1 + 1j * z) / (1 - 1j * z)
            return complex(-2j / (1 + z**2) * ratio**ell)
        return 0j
    if p == 1:
        return complex(z * fun_z_lp(z, ell, 0) - (1 if ell == 0 else 0))
    if p == 2:
        sign = int(np.sign(ell))
        return complex(z * fun_z_lp(z, ell, 1) + sign * ((-1) ** ell) * 1j)
    raise ValueError(f"Unsupported GPDF moment p={p}; expected 0, 1, or 2")

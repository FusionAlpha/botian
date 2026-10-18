from __future__ import annotations

"""McMillan-minimal realization of the finite 2025 BO-HH matrix.

This module is the Python counterpart of
``scripts/matlab/bo_hh2025_minimal_from_coeff.m``.  It preserves the finite
BO spectrum as a reduced dynamic pencil plus structural poles with certified
multiplicity.  It does not decide whether any root is physically meaningful.
"""

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import eigvals

from .hh_coefficients import HHMatrixCoefficients


def _readonly(values: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _pole_key(value: complex) -> tuple[int, int]:
    pair = np.asarray([value.real, value.imag], dtype=np.float64).view(np.uint64)
    return int(pair[0]), int(pair[1])


def _exact_pole_groups(poles: np.ndarray) -> tuple[np.ndarray, ...]:
    groups: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
    for index, pole in enumerate(np.asarray(poles, dtype=np.complex128)):
        groups.setdefault(_pole_key(complex(pole)), []).append(index)
    return tuple(np.asarray(indices, dtype=np.int64) for indices in groups.values())


@dataclass(frozen=True)
class HHMinimalGroup:
    pole: complex
    coefficient_indices: np.ndarray
    block_size: int
    rank: int
    known_pole_multiplicity: int
    singular_values: np.ndarray
    rank_threshold: float
    rank_gap: float
    rank_ambiguous: bool
    residue_frobenius_norm: float


@dataclass(frozen=True)
class HHMcMillanCertificate:
    schema_version: int
    grouping_mode: str
    original_size: int
    original_auxiliary_size: int
    field_size: int
    pole_group_count: int
    minimal_auxiliary_size: int
    minimal_size: int
    known_pole_count: int
    reconstructed_root_count: int
    count_closure: bool
    ambiguous_rank_count: int
    rank_tolerance: float
    mcmillan_minimal: bool


@dataclass(frozen=True)
class HHMinimalRealization:
    matrix: np.ndarray
    dynamic_poles: np.ndarray
    input_factor: np.ndarray
    output_factor: np.ndarray
    field_block: np.ndarray
    known_poles: np.ndarray
    groups: tuple[HHMinimalGroup, ...]
    certificate: HHMcMillanCertificate


@dataclass(frozen=True)
class HHMinimalSpectrum:
    roots: np.ndarray
    dynamic_roots: np.ndarray
    structural_poles: np.ndarray
    realization: HHMinimalRealization
    build_seconds: float
    eig_seconds: float


def _rank_diagnostics(
    singular_values: np.ndarray,
    rank_tolerance: float,
    *,
    rank_gap_tolerance: float = 100.0,
    ambiguity_factor: float = 10.0,
) -> tuple[int, float, float, bool]:
    values = np.asarray(singular_values, dtype=float)
    threshold = rank_tolerance * max(float(values[0]) if values.size else 0.0, 1.0)
    rank = int(np.count_nonzero(values > threshold))
    kept = 0.0 if rank == 0 else float(values[rank - 1])
    following = 0.0 if rank >= values.size else float(values[rank])
    rank_gap = float("inf") if following == 0.0 else max(
        kept, np.finfo(float).tiny
    ) / max(following, np.finfo(float).tiny)
    kept_near_threshold = rank > 0 and kept < threshold * ambiguity_factor
    following_near_threshold = (
        rank < values.size and following > threshold / ambiguity_factor
    )
    gap_ambiguous = rank > 0 and rank < values.size and rank_gap < rank_gap_tolerance
    zero_ambiguous = rank == 0 and values.size > 0 and values[0] > threshold / ambiguity_factor
    return (
        rank,
        threshold,
        rank_gap,
        bool(
            kept_near_threshold
            or gap_ambiguous
            or zero_ambiguous
            or (following_near_threshold and gap_ambiguous)
        ),
    )


def _field_block(coefficients: HHMatrixCoefficients) -> np.ndarray:
    species = int(coefficients.s_count)
    field_size = 3 * species + 6
    field = np.zeros((field_size, field_size), dtype=np.complex128)
    electric = 3 * species
    ex, ey, ez, bx, by, bz = range(electric, electric + 6)
    for s in range(species):
        jsx, jsy, jsz = s, species + s, 2 * species + s
        field[jsx, ex : ez + 1] = coefficients.bs[s, 0, :]
        field[jsy, ex : ez + 1] = coefficients.bs[s, 1, :]
        field[jsz, ex : ez + 1] = coefficients.bs[s, 2, :]
        field[ex, jsx] = -1.0
        field[ey, jsy] = -1.0
        field[ez, jsz] = -1.0
    field[ex, by] = coefficients.c2 * coefficients.kz
    field[ey, bx] = -coefficients.c2 * coefficients.kz
    field[ey, bz] = coefficients.c2 * coefficients.kx
    field[ez, by] = -coefficients.c2 * coefficients.kx
    field[bx, ey] = -coefficients.kz
    field[by, ex] = coefficients.kz
    field[by, ez] = -coefficients.kx
    field[bz, ey] = coefficients.kx
    return field


def build_hh_minimal_realization(
    coefficients: HHMatrixCoefficients,
    *,
    rank_tolerance: float = 3.0 * np.finfo(float).eps,
) -> HHMinimalRealization:
    """Build a reduced realization and a structural-pole multiplicity certificate."""

    coefficients.validate()
    if coefficients.iem != 1:
        raise NotImplementedError("HH minimal realization currently supports iem=1 only")
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")

    species = int(coefficients.s_count)
    field_size = 3 * species + 6
    electric = slice(3 * species, 3 * species + 3)
    factor_records: list[tuple[complex, np.ndarray, np.ndarray]] = []
    known_chunks: list[np.ndarray] = []
    group_records: list[HHMinimalGroup] = []
    ambiguous_count = 0

    for indices in _exact_pole_groups(coefficients.csnj):
        pole = complex(coefficients.csnj[int(indices[0])])
        residue = -np.sum(coefficients.bsnj[indices], axis=0)
        left, singular_values, right_h = np.linalg.svd(residue, full_matrices=False)
        rank, threshold, rank_gap, ambiguous = _rank_diagnostics(
            singular_values, rank_tolerance
        )
        roots = np.sqrt(singular_values[:rank])
        input_factor = np.zeros((rank, field_size), dtype=np.complex128)
        output_factor = np.zeros((field_size, rank), dtype=np.complex128)
        if rank:
            input_factor[:, electric] = roots[:, None] * right_h[:rank, :]
            output_factor[electric, :] = left[:, :rank] * roots[None, :]
        block_size = 3 * len(indices)
        known_multiplicity = block_size - rank
        if known_multiplicity < 0:
            raise AssertionError("HH residue rank exceeds the original pole block size")
        if known_multiplicity:
            known_chunks.append(np.full(known_multiplicity, pole, dtype=np.complex128))
        factor_records.append((pole, input_factor, output_factor))
        ambiguous_count += int(ambiguous)
        group_records.append(
            HHMinimalGroup(
                pole=pole,
                coefficient_indices=_readonly(indices, dtype=np.int64),
                block_size=block_size,
                rank=rank,
                known_pole_multiplicity=known_multiplicity,
                singular_values=_readonly(singular_values, dtype=float),
                rank_threshold=threshold,
                rank_gap=rank_gap,
                rank_ambiguous=ambiguous,
                residue_frobenius_norm=float(np.linalg.norm(residue, ord="fro")),
            )
        )

    minimal_auxiliary_size = sum(item[1].shape[0] for item in factor_records)
    dynamic_poles = np.empty(minimal_auxiliary_size, dtype=np.complex128)
    input_factor = np.zeros((minimal_auxiliary_size, field_size), dtype=np.complex128)
    output_factor = np.zeros((field_size, minimal_auxiliary_size), dtype=np.complex128)
    offset = 0
    for pole, input_block, output_block in factor_records:
        rank = input_block.shape[0]
        if not rank:
            continue
        slots = slice(offset, offset + rank)
        dynamic_poles[slots] = pole
        input_factor[slots] = input_block
        output_factor[:, slots] = output_block
        offset += rank
    if offset != minimal_auxiliary_size:
        raise AssertionError("HH minimal auxiliary-state accounting failed")

    field = _field_block(coefficients)
    minimal_size = minimal_auxiliary_size + field_size
    matrix = np.zeros((minimal_size, minimal_size), dtype=np.complex128)
    if minimal_auxiliary_size:
        indices = np.arange(minimal_auxiliary_size)
        matrix[indices, indices] = dynamic_poles
        matrix[:minimal_auxiliary_size, minimal_auxiliary_size:] = input_factor
        matrix[minimal_auxiliary_size:, :minimal_auxiliary_size] = output_factor
    matrix[minimal_auxiliary_size:, minimal_auxiliary_size:] = field
    known_poles = (
        np.concatenate(known_chunks) if known_chunks else np.empty(0, dtype=np.complex128)
    )
    reconstructed_count = minimal_size + int(known_poles.size)
    count_closure = reconstructed_count == int(coefficients.nn)
    certificate = HHMcMillanCertificate(
        schema_version=1,
        grouping_mode="exact-bitwise",
        original_size=int(coefficients.nn),
        original_auxiliary_size=3 * int(coefficients.snj),
        field_size=field_size,
        pole_group_count=len(group_records),
        minimal_auxiliary_size=minimal_auxiliary_size,
        minimal_size=minimal_size,
        known_pole_count=int(known_poles.size),
        reconstructed_root_count=reconstructed_count,
        count_closure=count_closure,
        ambiguous_rank_count=ambiguous_count,
        rank_tolerance=float(rank_tolerance),
        mcmillan_minimal=count_closure and ambiguous_count == 0,
    )
    return HHMinimalRealization(
        matrix=_readonly(matrix, dtype=np.complex128),
        dynamic_poles=_readonly(dynamic_poles, dtype=np.complex128),
        input_factor=_readonly(input_factor, dtype=np.complex128),
        output_factor=_readonly(output_factor, dtype=np.complex128),
        field_block=_readonly(field, dtype=np.complex128),
        known_poles=_readonly(known_poles, dtype=np.complex128),
        groups=tuple(group_records),
        certificate=certificate,
    )


def solve_hh_minimal(
    coefficients: HHMatrixCoefficients,
    *,
    rank_tolerance: float = 3.0 * np.finfo(float).eps,
) -> HHMinimalSpectrum:
    started = perf_counter()
    realization = build_hh_minimal_realization(
        coefficients, rank_tolerance=rank_tolerance
    )
    build_seconds = perf_counter() - started
    eig_started = perf_counter()
    dynamic_roots = eigvals(
        np.array(realization.matrix, copy=True), overwrite_a=True, check_finite=False
    )
    eig_seconds = perf_counter() - eig_started
    roots = np.concatenate((dynamic_roots, realization.known_poles))
    if roots.size != coefficients.nn:
        raise AssertionError(
            f"HH minimal root count {roots.size} does not close to {coefficients.nn}"
        )
    return HHMinimalSpectrum(
        roots=_readonly(roots, dtype=np.complex128),
        dynamic_roots=_readonly(dynamic_roots, dtype=np.complex128),
        structural_poles=realization.known_poles,
        realization=realization,
        build_seconds=float(build_seconds),
        eig_seconds=float(eig_seconds),
    )

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

import numpy as np
from scipy.linalg import eigvals

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
    LegacyMinimalRealization,
    LegacyMinimalSpectrum,
    LegacyScanPoint,
)


def _readonly(values: np.ndarray, *, dtype=None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _pole_key(value: complex) -> tuple[int, int]:
    pair = np.asarray([value.real, value.imag], dtype=np.float64).view(np.uint64)
    return int(pair[0]), int(pair[1])


def _exact_pole_groups(poles: np.ndarray) -> tuple[np.ndarray, ...]:
    groups: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
    for index, pole in enumerate(poles):
        groups.setdefault(_pole_key(complex(pole)), []).append(index)
    return tuple(np.asarray(indices, dtype=np.int64) for indices in groups.values())


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
    rank_gap = float("inf") if following == 0.0 else max(kept, np.finfo(float).tiny) / max(
        following,
        np.finfo(float).tiny,
    )
    near_threshold = bool(
        np.any((values > threshold / ambiguity_factor) & (values < threshold * ambiguity_factor))
    )
    gap_ambiguous = rank > 0 and rank < values.size and rank_gap < rank_gap_tolerance
    zero_ambiguous = rank == 0 and values.size > 0 and values[0] > threshold / ambiguity_factor
    return rank, threshold, rank_gap, near_threshold or gap_ambiguous or zero_ambiguous


def _es_field_block(coefficients: LegacyCoefficients) -> np.ndarray:
    if coefficients.es_residues is None:
        raise ValueError("ES minimal realization requires scalar residues")
    return np.asarray([[-np.sum(coefficients.es_residues)]], dtype=np.complex128)


def _em_field_block(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    coefficients: LegacyCoefficients,
) -> np.ndarray:
    if coefficients.em_constant is None:
        raise ValueError("EM minimal realization requires the constant current block")
    _, kx, kz = effective_em_wavevector(point)
    field = np.zeros((9, 9), dtype=np.complex128)
    field[0:3, 3:6] = coefficients.em_constant
    field[3, 0] = -1.0
    field[4, 1] = -1.0
    field[5, 2] = -1.0
    c2 = ctx.constants.c2
    field[3, 7] = c2 * kz
    field[4, 6] = -c2 * kz
    field[4, 8] = c2 * kx
    field[5, 7] = -c2 * kx
    field[6, 4] = -kz
    field[7, 3] = kz
    field[7, 5] = -kx
    field[8, 4] = kx
    return field


def build_minimal_realization(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    coefficients: LegacyCoefficients | None = None,
    rank_tolerance: float = 1e-12,
) -> LegacyMinimalRealization:
    """Build the exact finite-model McMillan realization from BO coefficients."""

    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")
    if ctx.case.model.iem == 2:
        raise NotImplementedError("The minimal backend does not reduce the optional Darwin pencil")
    if ctx.case.model.iem == 0:
        coeff = coefficients or evaluate_es3d_coefficients(ctx, point)
        component_multiplicity = 1
        field_size = 1
        field = _es_field_block(coeff)
    elif ctx.case.model.iem == 1:
        coeff = coefficients or evaluate_em3d_coefficients(ctx, point)
        component_multiplicity = 3
        field_size = 9
        field = _em_field_block(ctx, point, coeff)
    else:
        raise ValueError(f"Unsupported legacy iem={ctx.case.model.iem}")

    grouped_indices = _exact_pole_groups(coeff.poles)
    group_records: list[LegacyMinimalGroup] = []
    factor_records: list[tuple[complex, np.ndarray, np.ndarray]] = []
    known_chunks: list[np.ndarray] = []
    ambiguous_count = 0

    for indices in grouped_indices:
        pole = complex(coeff.poles[int(indices[0])])
        block_size = component_multiplicity * len(indices)
        if ctx.case.model.iem == 0:
            assert coeff.es_residues is not None
            residue = -np.sum(coeff.poles[indices] * coeff.es_residues[indices])
            singular_values = np.asarray([abs(residue)], dtype=float)
            rank, threshold, rank_gap, ambiguous = _rank_diagnostics(
                singular_values,
                rank_tolerance,
            )
            if rank:
                input_factor = np.ones((1, 1), dtype=np.complex128)
                output_factor = np.asarray([[residue]], dtype=np.complex128)
            else:
                input_factor = np.zeros((0, 1), dtype=np.complex128)
                output_factor = np.zeros((1, 0), dtype=np.complex128)
            residue_norm = float(abs(residue))
        else:
            assert coeff.em_residues is not None
            residue3 = -np.sum(coeff.em_residues[indices], axis=0)
            left, singular_values, right_h = np.linalg.svd(residue3, full_matrices=False)
            rank, threshold, rank_gap, ambiguous = _rank_diagnostics(
                singular_values,
                rank_tolerance,
            )
            roots = np.sqrt(singular_values[:rank])
            input_factor = np.zeros((rank, field_size), dtype=np.complex128)
            output_factor = np.zeros((field_size, rank), dtype=np.complex128)
            if rank:
                input_factor[:, 3:6] = roots[:, None] * right_h[:rank, :]
                output_factor[3:6, :] = left[:, :rank] * roots[None, :]
            residue_norm = float(np.linalg.norm(residue3, ord="fro"))

        known_multiplicity = block_size - rank
        if known_multiplicity < 0:
            raise AssertionError("Residue rank exceeds its original pole block size")
        if known_multiplicity:
            known_chunks.append(np.full(known_multiplicity, pole, dtype=np.complex128))
        factor_records.append((pole, input_factor, output_factor))
        ambiguous_count += int(ambiguous)
        group_records.append(
            LegacyMinimalGroup(
                pole=pole,
                coefficient_indices=_readonly(indices, dtype=np.int64),
                block_size=block_size,
                rank=rank,
                known_pole_multiplicity=known_multiplicity,
                singular_values=_readonly(singular_values, dtype=float),
                rank_threshold=threshold,
                rank_gap=rank_gap,
                rank_ambiguous=ambiguous,
                residue_frobenius_norm=residue_norm,
            )
        )

    minimal_auxiliary_size = sum(record[1].shape[0] for record in factor_records)
    input_factor = np.zeros((minimal_auxiliary_size, field_size), dtype=np.complex128)
    output_factor = np.zeros((field_size, minimal_auxiliary_size), dtype=np.complex128)
    dynamic_poles = np.empty(minimal_auxiliary_size, dtype=np.complex128)
    offset = 0
    for pole, input_block, output_block in factor_records:
        rank = input_block.shape[0]
        if rank:
            slots = slice(offset, offset + rank)
            dynamic_poles[slots] = pole
            input_factor[slots] = input_block
            output_factor[:, slots] = output_block
            offset += rank
    if offset != minimal_auxiliary_size:
        raise AssertionError("Minimal auxiliary state accounting failed")

    minimal_size = minimal_auxiliary_size + field_size
    matrix = np.zeros((minimal_size, minimal_size), dtype=np.complex128)
    if minimal_auxiliary_size:
        matrix[np.arange(minimal_auxiliary_size), np.arange(minimal_auxiliary_size)] = dynamic_poles
        matrix[:minimal_auxiliary_size, minimal_auxiliary_size:] = input_factor
        matrix[minimal_auxiliary_size:, :minimal_auxiliary_size] = output_factor
    matrix[minimal_auxiliary_size:, minimal_auxiliary_size:] = field
    known_poles = (
        np.concatenate(known_chunks) if known_chunks else np.empty(0, dtype=np.complex128)
    )
    original_auxiliary_size = component_multiplicity * ctx.snj
    reconstructed_count = minimal_size + known_poles.size
    count_closure = reconstructed_count == ctx.nn
    certificate = LegacyMcMillanCertificate(
        grouping_mode="exact-bitwise",
        original_size=ctx.nn,
        original_auxiliary_size=original_auxiliary_size,
        field_size=field_size,
        pole_group_count=len(group_records),
        minimal_auxiliary_size=minimal_auxiliary_size,
        minimal_size=minimal_size,
        known_pole_count=int(known_poles.size),
        reconstructed_root_count=reconstructed_count,
        count_closure=count_closure,
        ambiguous_rank_count=ambiguous_count,
        rank_tolerance=rank_tolerance,
        mcmillan_minimal=count_closure and ambiguous_count == 0,
    )
    return LegacyMinimalRealization(
        matrix=_readonly(matrix, dtype=np.complex128),
        dynamic_poles=_readonly(dynamic_poles, dtype=np.complex128),
        input_factor=_readonly(input_factor, dtype=np.complex128),
        output_factor=_readonly(output_factor, dtype=np.complex128),
        field_block=_readonly(field, dtype=np.complex128),
        known_poles=_readonly(known_poles, dtype=np.complex128),
        groups=tuple(group_records),
        certificate=certificate,
    )


def solve_minimal_point(
    ctx: LegacyContext,
    point: LegacyScanPoint,
    *,
    rank_tolerance: float = 1e-12,
) -> LegacyMinimalSpectrum:
    started = perf_counter()
    realization = build_minimal_realization(
        ctx,
        point,
        rank_tolerance=rank_tolerance,
    )
    build_seconds = perf_counter() - started
    eig_started = perf_counter()
    dynamic_roots = eigvals(
        np.array(realization.matrix, copy=True),
        overwrite_a=True,
        check_finite=False,
    )
    eig_seconds = perf_counter() - eig_started
    roots = np.concatenate((dynamic_roots, realization.known_poles))
    if roots.size != ctx.nn:
        raise AssertionError(f"Minimal root count {roots.size} does not close to {ctx.nn}")
    return LegacyMinimalSpectrum(
        roots=_readonly(roots, dtype=np.complex128),
        dynamic_roots=_readonly(dynamic_roots, dtype=np.complex128),
        structural_poles=realization.known_poles,
        realization=realization,
        build_seconds=build_seconds,
        eig_seconds=eig_seconds,
    )

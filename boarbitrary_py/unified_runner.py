from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .case_registry import BOCaseRecord, get_case_record
from .hh_cases import initialize_hh_case
from .hh_coefficients import compute_hh_coefficients
from .hh_minimal import solve_hh_minimal
from .legacy_pdrk.initialize import initialize_legacy_case
from .legacy_pdrk.io import load_legacy_case
from .legacy_pdrk.scan import iter_legacy_scan_points
from .legacy_pdrk.solver import solve_legacy_scan
from .scan import iter_scan_points as iter_hh_scan_points
from .scan import run_scan as run_hh_scan
from .solver import sort_by_growth


@dataclass(frozen=True)
class InitializedBOCase:
    record: BOCaseRecord
    context: Any


@dataclass(frozen=True)
class UnifiedScanResult:
    schema_version: int
    case_id: str
    source_family: str
    backend: str
    roots: np.ndarray
    normalized_roots: np.ndarray
    scan_indices: np.ndarray
    pa: np.ndarray
    pb: np.ndarray
    k: np.ndarray
    kx: np.ndarray
    kz: np.ndarray
    k_normalized: np.ndarray
    kx_normalized: np.ndarray
    kz_normalized: np.ndarray
    theta_deg: np.ndarray
    matrix_size: int
    canonical_scan_points: int
    normalization_frequency: float
    normalization_wavenumber: float
    elapsed_seconds: float
    reduced_matrix_sizes: np.ndarray
    structural_root_counts: np.ndarray
    model_parameters: dict[str, Any]
    overrides: tuple[dict[str, Any], ...]
    validation_status: dict[str, Any]

    @property
    def is_full_scan(self) -> bool:
        return self.scan_indices.size == self.canonical_scan_points

    def manifest(self, *, npz_file: str | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "source_family": self.source_family,
            "backend": self.backend,
            "model_parameters": self.model_parameters,
            "matrix_dimension": self.matrix_size,
            "roots_per_scan": int(self.roots.shape[1]),
            "normalization": {
                "frequency": self.normalization_frequency,
                "wavenumber": self.normalization_wavenumber,
            },
            "scan": {
                "canonical_points": self.canonical_scan_points,
                "selected_points": int(self.scan_indices.size),
                "selected_indices_0based": self.scan_indices.tolist(),
                "is_full_scan": self.is_full_scan,
                "coordinate_arrays": [
                    "pa",
                    "pb",
                    "k",
                    "kx",
                    "kz",
                    "k_normalized",
                    "kx_normalized",
                    "kz_normalized",
                    "theta_deg",
                ],
            },
            "timing": {"solve_seconds": self.elapsed_seconds},
            "overrides": list(self.overrides),
            "validation_status": self.validation_status,
            "reference_data_used_by_solver": False,
        }
        if npz_file is not None:
            payload["npz_file"] = npz_file
        return payload


def default_workspace() -> Path:
    return Path(__file__).resolve().parents[1]


def initialize_registered_case(
    case_id: str,
    *,
    workspace: str | Path | None = None,
) -> InitializedBOCase:
    record = get_case_record(case_id)
    root = (Path(workspace) if workspace is not None else default_workspace()).resolve()
    if record.source_family == "2018_bo_pdrk":
        context = initialize_legacy_case(
            load_legacy_case(case_id, cases_root=root / "legacy_cases")
        )
    elif record.source_family == "2025_bo_arbitrary_hh":
        context = initialize_hh_case(case_id, cases_root=root / "paper_cases")
    else:
        raise ValueError(f"Unsupported source family {record.source_family!r}")
    _validate_initialized_case(record, context)
    return InitializedBOCase(record=record, context=context)


def run_registered_case(
    case_id: str,
    *,
    workspace: str | Path | None = None,
    scan_indices: list[int] | tuple[int, ...] | np.ndarray | None = None,
    backend: str = "auto",
    quadrature_order: int = 80,
) -> UnifiedScanResult:
    initialized = initialize_registered_case(case_id, workspace=workspace)
    record = initialized.record
    ctx = initialized.context
    selected_backend = record.default_backend if backend == "auto" else backend
    if selected_backend not in record.supported_backends:
        raise ValueError(
            f"Backend {selected_backend!r} is not supported by {case_id}; "
            f"choices: {record.supported_backends}"
        )
    selected = _validated_indices(scan_indices, record.scan.points)

    if record.source_family == "2018_bo_pdrk":
        raw = solve_legacy_scan(ctx, selected, backend=selected_backend)
        points = tuple(iter_legacy_scan_points(ctx))
        chosen_points = tuple(points[int(index)] for index in selected)
        roots = np.asarray(raw.roots, dtype=np.complex128)
        pa = np.asarray([point.pa_value for point in chosen_points], dtype=float)
        pb = np.asarray([point.pb_value for point in chosen_points], dtype=float)
        k = np.asarray(raw.k, dtype=float)
        kx = np.asarray(raw.kx, dtype=float)
        kz = np.asarray(raw.kz, dtype=float)
        theta = np.asarray(raw.theta_deg, dtype=float)
        reduced_sizes = np.asarray(raw.reduced_matrix_sizes, dtype=np.int64)
        structural_counts = np.asarray(raw.structural_root_counts, dtype=np.int64)
        elapsed = float(raw.elapsed_seconds)
        species_count = int(ctx.s_count)
        b0 = float(ctx.case.b0)
    else:
        if selected_backend == "full":
            raw = run_hh_scan(
                ctx,
                scan_indices=selected,
                quadrature_order=quadrature_order,
            )
            roots = np.atleast_2d(np.asarray(raw.ww, dtype=np.complex128))
            pa = np.asarray(raw.pa_used, dtype=float)
            pb = np.asarray(raw.pb_used, dtype=float)
            k = np.asarray(raw.kk, dtype=float).reshape(-1)
            kx = np.asarray(raw.kxx, dtype=float).reshape(-1)
            kz = np.asarray(raw.kzz, dtype=float).reshape(-1)
            theta = np.asarray(raw.tt, dtype=float).reshape(-1)
            reduced_sizes = np.full(len(selected), record.matrix_size, dtype=np.int64)
            structural_counts = np.zeros(len(selected), dtype=np.int64)
            elapsed = float(raw.elapsed_seconds)
        elif selected_backend == "minimal":
            (
                roots,
                pa,
                pb,
                k,
                kx,
                kz,
                theta,
                reduced_sizes,
                structural_counts,
                elapsed,
            ) = _run_hh_minimal_scan(
                ctx,
                selected,
                quadrature_order=quadrature_order,
            )
        else:
            raise ValueError(f"Unsupported HH backend: {selected_backend}")
        species_count = int(ctx.s_count)
        b0 = float(ctx.setup.b0)

    if roots.shape != (len(selected), record.roots_per_scan):
        raise AssertionError(
            f"Unexpected root shape for {case_id}: {roots.shape}; "
            f"expected {(len(selected), record.roots_per_scan)}"
        )
    wn = float(ctx.wn)
    kn = float(ctx.kn)
    return UnifiedScanResult(
        schema_version=1,
        case_id=case_id,
        source_family=record.source_family,
        backend=selected_backend,
        roots=roots,
        normalized_roots=roots / wn,
        scan_indices=selected,
        pa=pa,
        pb=pb,
        k=k,
        kx=kx,
        kz=kz,
        k_normalized=k / kn,
        kx_normalized=kx / kn,
        kz_normalized=kz / kn,
        theta_deg=theta,
        matrix_size=record.matrix_size,
        canonical_scan_points=record.scan.points,
        normalization_frequency=wn,
        normalization_wavenumber=kn,
        elapsed_seconds=elapsed,
        reduced_matrix_sizes=reduced_sizes,
        structural_root_counts=structural_counts,
        model_parameters={
            **asdict(record.model),
            "species_count": species_count,
            "b0_tesla": b0,
            "quadrature_order": quadrature_order if record.source_family == "2025_bo_arbitrary_hh" else None,
        },
        overrides=tuple(asdict(override) for override in record.overrides),
        validation_status=asdict(record.validation),
    )


def parse_scan_selection(selection: str, scan_points: int) -> np.ndarray:
    normalized = selection.strip().lower()
    if normalized == "all":
        return np.arange(scan_points, dtype=np.int64)
    aliases = {"first": 0, "mid": scan_points // 2, "last": scan_points - 1}
    values: list[int] = []
    for token in (part.strip().lower() for part in selection.split(",")):
        if not token:
            continue
        if token in aliases:
            values.append(aliases[token])
        else:
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(
                    "Scan selection must be 'all' or comma-separated integers/first/mid/last"
                ) from exc
    return _validated_indices(values, scan_points)


def save_unified_result(
    result: UnifiedScanResult,
    output_dir: str | Path,
    *,
    stem: str | None = None,
) -> tuple[Path, Path]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_stem = stem or _result_stem(result)
    npz_path = destination / f"{output_stem}.npz"
    json_path = destination / f"{output_stem}.json"
    np.savez_compressed(
        npz_path,
        schema_version=np.asarray(result.schema_version, dtype=np.int64),
        case_id=np.asarray(result.case_id),
        source_family=np.asarray(result.source_family),
        backend=np.asarray(result.backend),
        roots=result.roots,
        normalized_roots=result.normalized_roots,
        scan_indices=result.scan_indices,
        pa=result.pa,
        pb=result.pb,
        k=result.k,
        kx=result.kx,
        kz=result.kz,
        k_normalized=result.k_normalized,
        kx_normalized=result.kx_normalized,
        kz_normalized=result.kz_normalized,
        theta_deg=result.theta_deg,
        matrix_size=np.asarray(result.matrix_size, dtype=np.int64),
        canonical_scan_points=np.asarray(result.canonical_scan_points, dtype=np.int64),
        normalization_frequency=np.asarray(result.normalization_frequency, dtype=float),
        normalization_wavenumber=np.asarray(result.normalization_wavenumber, dtype=float),
        elapsed_seconds=np.asarray(result.elapsed_seconds, dtype=float),
        reduced_matrix_sizes=result.reduced_matrix_sizes,
        structural_root_counts=result.structural_root_counts,
    )
    manifest = result.manifest(npz_file=npz_path.name)
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return npz_path, json_path


def _validated_indices(
    scan_indices: list[int] | tuple[int, ...] | np.ndarray | None,
    scan_points: int,
) -> np.ndarray:
    if scan_indices is None:
        return np.arange(scan_points, dtype=np.int64)
    selected = np.asarray(scan_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("scan_indices must be a nonempty one-dimensional sequence")
    invalid = selected[(selected < 0) | (selected >= scan_points)]
    if invalid.size:
        raise ValueError(f"scan_indices out of range for {scan_points} points: {invalid.tolist()}")
    if np.unique(selected).size != selected.size:
        raise ValueError(f"scan_indices contains duplicates: {selected.tolist()}")
    return selected


def _validate_initialized_case(record: BOCaseRecord, context: Any) -> None:
    matrix_size = int(context.nn)
    roots_per_scan = int(context.nw)
    scan_points = len(context.pa) * len(context.pb)
    actual = (matrix_size, roots_per_scan, scan_points)
    expected = (record.matrix_size, record.roots_per_scan, record.scan.points)
    if actual != expected:
        raise ValueError(
            f"Registry/context mismatch for {record.case_id}: actual={actual}, expected={expected}"
        )


def _result_stem(result: UnifiedScanResult) -> str:
    if result.is_full_scan:
        scope = "all"
    elif result.scan_indices.size <= 8:
        scope = "indices-" + "-".join(str(int(value)) for value in result.scan_indices)
    else:
        digest = hashlib.sha256(result.scan_indices.tobytes()).hexdigest()[:12]
        scope = f"subset-{result.scan_indices.size}-{digest}"
    return f"{result.case_id}__{result.backend}__{scope}"


def _run_hh_minimal_scan(
    ctx: Any,
    selected: np.ndarray,
    *,
    quadrature_order: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    points = tuple(iter_hh_scan_points(ctx, scan_indices=selected))
    roots = np.empty((len(points), ctx.nw), dtype=np.complex128)
    pa = np.empty(len(points), dtype=float)
    pb = np.empty(len(points), dtype=float)
    k = np.empty(len(points), dtype=float)
    kx = np.empty(len(points), dtype=float)
    kz = np.empty(len(points), dtype=float)
    theta = np.empty(len(points), dtype=float)
    reduced_sizes = np.empty(len(points), dtype=np.int64)
    structural_counts = np.empty(len(points), dtype=np.int64)
    started = perf_counter()
    for row, point in enumerate(points):
        coefficients = compute_hh_coefficients(
            ctx, point, quadrature_order=quadrature_order
        )
        spectrum = solve_hh_minimal(coefficients)
        roots[row] = sort_by_growth(spectrum.roots)
        pa[row] = point.pa_value
        pb[row] = point.pb_value
        k[row] = point.k
        kx[row] = point.kx
        kz[row] = point.kz
        theta[row] = point.theta_deg
        reduced_sizes[row] = spectrum.realization.certificate.minimal_size
        structural_counts[row] = spectrum.realization.certificate.known_pole_count
    return (
        roots,
        pa,
        pb,
        k,
        kx,
        kz,
        theta,
        reduced_sizes,
        structural_counts,
        perf_counter() - started,
    )

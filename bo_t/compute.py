"""Recompute the seven figure datasets from bundled numerical inputs.

This module never reads frozen plotting data or reference branches. Each scan
performs one reduced source solve, restores its structural poles, then passes
that single source row to the unchanged local selector.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from boarbitrary_py.selector_artifacts import (
    freeze_prediction_artifact,
    verify_prediction_artifact,
)
from boarbitrary_py.selector_legacy import select_legacy_case
from boarbitrary_py.selector_profiles import get_selector_profile
from boarbitrary_py.selector_select import select_hh_case
from boarbitrary_py.unified_runner import run_registered_case


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = frozenset({"declared_auxiliary_pole", "zero_or_constraint_root"})
SELECTED = frozenset({"core_stable", "near_stable_only"})


@dataclass(frozen=True)
class Leg:
    name: str
    input_name: str
    case_id: str
    scan_count: int
    root_count: int
    x_coordinate: str = "k_normalized"


@dataclass(frozen=True)
class Recipe:
    legs: tuple[Leg, ...]
    family: str
    quadrature_order: int
    audit_all_roots: bool


RECIPES = {
    "es1d_beam": Recipe((Leg("scan", "es1d_beam", "bo2018_es1d_beam", 50, 49),), "legacy", 120, False),
    "umeda_ring_beam": Recipe((Leg("scan", "umeda_ring_beam", "bo2018_ringbeam_umeda12", 117, 825),), "legacy", 120, False),
    "signed_em": Recipe((
        Leg("negative", "signed_em_negative", "bo2018_darwin", 250, 153, "kz_normalized"),
        Leg("positive", "signed_em_positive", "bo2018_darwin", 250, 153, "kz_normalized"),
    ), "legacy", 80, True),
    "hh_ring_beam": Recipe((Leg("scan", "hh_ring_beam", "bo2025_ringbeam_umeda12_theta40", 117, 948),), "hh", 80, False),
    "firehose": Recipe((Leg("scan", "firehose", "bo2025_firehose_astfalk17_theta45", 46, 732),), "hh", 80, True),
    "muschietti_beam": Recipe((Leg("scan", "muschietti_beam", "bo2018_muschietti17", 131, 2961),), "legacy", 120, False),
    "loss_cone_mirror": Recipe((Leg("scan", "loss_cone_mirror", "bo2018_loss_cone_mirror", 100, 345, "kz_normalized"),), "legacy", 80, True),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {path.name}\n", encoding="ascii"
    )


def _read_json(path: Path) -> dict:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = f"{_sha256(path)}  {path.name}"
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != expected:
        raise ValueError(f"Checksum mismatch or missing commit: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _profile(case_name: str, leg: Leg):
    base = get_selector_profile(leg.case_id)
    if case_name != "hh_ring_beam":
        return base
    # Preserve the active printed-fit development policy, not the older J8
    # fallback. This does not promote a development result to held-out evidence.
    return replace(
        base,
        profile_id=f"{leg.case_id}__forward-j24-development",
        fallback=replace(base.fallback, discretization=replace(base.fallback.discretization, pole_order=24)),
        scan_scope=replace(
            base.scan_scope, source_variant="printed_fit_reconstruction_raw",
            partition="posthoc_development_full", held_out=False,
        ),
        provenance=(
            "Forward-J24 fallback was chosen after inspecting archive misses. "
            "This reproduces that development policy; predictions do not read references."
        ),
        metadata=tuple((key, value) for key, value in base.metadata if key not in {
            "declared_before_evaluation", "evaluation_labels_read", "j_probe_rationale", "profile_basis",
        }) + (
            ("declared_before_evaluation", False),
            ("evaluation_labels_read", True),
            ("profile_basis", "posthoc_method_development_after_archive_inspection"),
            ("j_probe_rationale", "Forward-J24 fallback after development-set archive inspection"),
        ),
    )


def _code_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "boarbitrary_py").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _input_hashes(recipe: Recipe) -> dict[str, str]:
    hashes = {}
    for leg in recipe.legs:
        capsule = ROOT / "inputs" / leg.input_name
        paths = sorted(path for path in capsule.rglob("*") if path.is_file())
        if not paths:
            raise FileNotFoundError(f"Missing bundled numerical inputs: {capsule}")
        hashes.update({path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths})
    return hashes


def _environment() -> dict:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": sys.platform,
        "machine": platform.machine(), "blas_threads": 1,
        "blas_libraries": [
            {key: item.get(key) for key in ("internal_api", "version", "threading_layer", "architecture")}
            for item in threadpool_info() if item.get("user_api") == "blas"
        ],
    }


def _plan(case_name: str, recipe: Recipe, smoke: bool) -> dict:
    legs = []
    for leg in recipe.legs:
        profile = _profile(case_name, leg)
        legs.append({
            "name": leg.name, "input_capsule": f"inputs/{leg.input_name}",
            "case_id": leg.case_id, "registered_scan_count": leg.scan_count,
            "expected_roots_per_scan": leg.root_count,
            "scan_indices": [leg.scan_count // 2] if smoke else list(range(leg.scan_count)),
            "x_coordinate": leg.x_coordinate, "profile": profile.to_dict(),
        })
    return {
        "schema_version": 1, "case": case_name,
        "mode": "smoke" if smoke else "recompute",
        "full_scan": not smoke, "legs": legs,
        "source_backend": "native_python_minimal_with_structural_pole_restoration",
        "quadrature_order": recipe.quadrature_order, "audit_all_roots": recipe.audit_all_roots,
        "include_core": True, "stability_tolerance": 1e-3,
        "inputs": _input_hashes(recipe), "code": _code_hashes(), "environment": _environment(),
        "reference_data_read": False, "frozen_plotting_data_read": False,
        "hh_ring_beam_policy_is_posthoc_development": case_name == "hh_ring_beam",
        "source_parity_with_historical_backend": "not_asserted_by_this_run",
        "selected_roots_are": "finite_discretization_stable_candidates_not_certified_physical_modes",
        "excluded_from_gray_layer": sorted(EXCLUDED), "selected_states": sorted(SELECTED),
    }


def _read_scan(directory: Path, leg: Leg, index: int, fingerprint: str):
    commit = _read_json(directory / "scan_commit.json")
    if (commit["plan_sha256"] != fingerprint or commit["scan_index"] != index
            or commit["case_id"] != leg.case_id or commit["leg"] != leg.name):
        raise ValueError(f"Stale or mismatched scan commit: {directory}")
    for name, expected in commit["files"].items():
        if Path(name).name != name or _sha256(directory / name) != expected:
            raise ValueError(f"Scan artifact changed: {directory / name}")
    source = _read_json(directory / "source.json")
    verified = verify_prediction_artifact(directory / "prediction.json")
    manifest = verified.manifest
    if (manifest.source.case_id != leg.case_id or manifest.source.scan_index != index
            or manifest.source.root_count != leg.root_count
            or manifest.source.artifact_path != "source.npz"
            or dict(manifest.source.metadata).get("source_artifact_sha256") != _sha256(directory / "source.npz")):
        raise ValueError(f"Prediction is not linked to the committed source: {directory}")
    with np.load(directory / "source.npz", allow_pickle=False) as stored:
        roots = np.asarray(stored["normalized_roots"], dtype=np.complex128)
        if roots.shape != (1, leg.root_count) or stored["scan_indices"].tolist() != [index]:
            raise ValueError(f"Invalid source shape/index: {directory}")
    ordered = sorted(manifest.roots, key=lambda row: row.raw_index)
    if len(ordered) != leg.root_count or any(row.raw_index != i for i, row in enumerate(ordered)):
        raise ValueError(f"Prediction does not close over the source roots: {directory}")
    if not np.array_equal(np.asarray([row.source_root for row in ordered]), roots[0]):
        raise ValueError(f"Prediction roots differ from the source artifact: {directory}")
    counts = Counter(row.final_state.value for row in ordered)
    if sum(counts.values()) != leg.root_count or dict(counts) != commit["state_counts"]:
        raise ValueError(f"State accounting mismatch: {directory}")
    x = float(source["x"])
    raw = [(x, row.source_root.real, row.source_root.imag) for row in ordered if row.final_state.value not in EXCLUDED]
    selected = [(x, row.source_root.real, row.source_root.imag) for row in ordered if row.final_state.value in SELECTED]
    return np.asarray(raw, dtype=float).reshape(-1, 3), np.asarray(selected, dtype=float).reshape(-1, 3), commit


def _compute_scan(case_name: str, recipe: Recipe, leg: Leg, index: int, directory: Path, fingerprint: str) -> None:
    workspace = ROOT / "inputs" / leg.input_name
    started = perf_counter()
    source = run_registered_case(
        leg.case_id, workspace=workspace, scan_indices=[index], backend="minimal",
        quadrature_order=recipe.quadrature_order,
    )
    source_seconds = perf_counter() - started
    roots = np.asarray(source.normalized_roots, dtype=np.complex128)
    if (roots.shape != (1, leg.root_count) or not np.all(np.isfinite(roots))
            or int(source.reduced_matrix_sizes[0]) + int(source.structural_root_counts[0]) != leg.root_count):
        raise ValueError(f"Invalid reduced source/count accounting: {case_name}/{leg.name}/{index}")
    x = float(getattr(source, leg.x_coordinate)[0])
    source_path = directory / "source.npz"
    np.savez_compressed(
        source_path, normalized_roots=roots,
        scan_indices=np.asarray([index], dtype=np.int64), case_id=np.asarray(leg.case_id),
        source_role=np.asarray("fresh_native_python_minimal_source"),
        reduced_matrix_size=np.asarray(int(source.reduced_matrix_sizes[0])),
        structural_pole_count=np.asarray(int(source.structural_root_counts[0])),
    )
    _write_json(directory / "source.json", {
        "case_id": leg.case_id, "scan_index": index, "x": x,
        "x_coordinate": leg.x_coordinate, "root_count": leg.root_count,
        "reduced_matrix_size": int(source.reduced_matrix_sizes[0]),
        "structural_pole_count": int(source.structural_root_counts[0]),
        "source_sha256": _sha256(source_path), "source_seconds": source_seconds,
        "reference_data_read": False,
    })
    selector = select_hh_case if recipe.family == "hh" else select_legacy_case
    selected = selector(
        leg.case_id, scan_index=index, output_dir=directory, workspace=workspace,
        source_spectrum_path=source_path, profile=_profile(case_name, leg),
        stability_tolerance=1e-3, quadrature_order=recipe.quadrature_order,
        include_core=True, audit_all_roots=recipe.audit_all_roots, artifact_stem="prediction",
    )
    # The selector records an absolute path. Recommit only that provenance field
    # with a relative equivalent before creating this scan's release commit.
    # The root values, states, profile, and all evidence remain unchanged.
    portable = replace(selected.manifest, source=replace(selected.manifest.source, artifact_path="source.npz"))
    freeze_prediction_artifact(portable, directory, stem="prediction")
    verified = verify_prediction_artifact(directory / "prediction.json")
    counts = Counter(row.final_state.value for row in verified.manifest.roots)
    names = ("source.npz", "source.json", "source.json.sha256", "prediction.json", "prediction.csv", "prediction.sha256")
    _write_json(directory / "scan_commit.json", {
        "plan_sha256": fingerprint, "case_id": leg.case_id, "leg": leg.name,
        "scan_index": index, "state_counts": dict(counts),
        "files": {name: _sha256(directory / name) for name in names},
        "source_seconds": source_seconds, "scan_wall_seconds": perf_counter() - started,
        "all_root_audit_disagreement_count": selected.all_root_audit_disagreement_count,
        "source_path_base": "prediction_directory",
    })


def recompute(case_name: str, output_dir: Path, *, smoke: bool = False, resume: bool = False):
    """Return fresh ``(raw, selected, provenance)`` arrays for one figure.

    ``output_dir`` is the case-specific result directory. Resume accepts only
    committed scans from the same input, code, numerical policy and environment.
    An interrupted uncommitted temporary scan is never read as a result.
    """
    if case_name not in RECIPES:
        raise ValueError(f"Unknown case {case_name!r}; choose from {', '.join(RECIPES)}")
    recipe = RECIPES[case_name]
    destination = Path(output_dir).resolve()
    for protected in (ROOT / "inputs", ROOT / "boarbitrary_py", ROOT / "bo_t", ROOT / "data"):
        if destination == protected or destination.is_relative_to(protected):
            raise ValueError("Numerical output must not be written into bundled source or data")
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(f"Output is not empty: {destination}; use --resume or a new directory")
    started = perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        plan = _plan(case_name, recipe, smoke)
        fingerprint = _canonical_hash(plan)
        plan_path = destination / "run_plan.json"
        if plan_path.exists():
            if not resume or _read_json(plan_path) != plan:
                raise ValueError("Resume rejected: numerical inputs, code, settings or environment changed")
        else:
            if destination.exists() and any(destination.iterdir()):
                raise ValueError("Cannot resume a nonempty directory without a committed run plan")
            destination.mkdir(parents=True, exist_ok=True)
            _write_json(plan_path, plan)
        all_raw, all_selected, commits = [], [], []
        resumed = 0
        for leg, leg_plan in zip(recipe.legs, plan["legs"], strict=True):
            leg_directory = destination / leg.name
            leg_directory.mkdir(exist_ok=True)
            indices = leg_plan["scan_indices"]
            for position, index in enumerate(indices):
                scan_directory = leg_directory / f"scan-{index:04d}"
                was_resumed = scan_directory.exists()
                if was_resumed:
                    if not resume:
                        raise FileExistsError(scan_directory)
                    resumed += 1
                else:
                    stage = Path(tempfile.mkdtemp(prefix=f".scan-{index:04d}-", dir=leg_directory))
                    _compute_scan(case_name, recipe, leg, index, stage, fingerprint)
                    # Verify a complete scan before its directory becomes visible
                    # as resumable evidence. Failed work remains an uncommitted .scan.
                    _read_scan(stage, leg, index, fingerprint)
                    stage.rename(scan_directory)
                raw, selected, commit = _read_scan(scan_directory, leg, index, fingerprint)
                all_raw.append(raw)
                all_selected.append(selected)
                commits.append({
                    "path": (scan_directory / "scan_commit.json").relative_to(destination).as_posix(),
                    "sha256": _sha256(scan_directory / "scan_commit.json"),
                    "scan_index": index, "leg": leg.name,
                    "source_seconds": commit["source_seconds"],
                    "scan_wall_seconds": commit["scan_wall_seconds"],
                })
                action = "verified resume" if was_resumed else "computed"
                print(f"{case_name}/{leg.name}: {position + 1}/{len(indices)} scan {index}, {len(selected)} selected ({action})", flush=True)
        if _code_hashes() != plan["code"] or _input_hashes(recipe) != plan["inputs"]:
            raise RuntimeError("Code or numerical inputs changed during the run; results were not finalized")
        raw = np.concatenate(all_raw, axis=0)
        selected = np.concatenate(all_selected, axis=0)
        arrays_path = destination / "arrays.npz"
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".arrays-", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, raw=raw, selected=selected)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, arrays_path)
        provenance = {
            **plan, "status": "complete", "plan_sha256": fingerprint,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "actual_scan_count": len(commits), "registered_scan_count": sum(leg.scan_count for leg in recipe.legs),
            "scans": commits, "resumed_scan_count": resumed,
            "invocation_seconds": perf_counter() - started,
            "accumulated_scan_seconds": sum(item["scan_wall_seconds"] for item in commits),
            "raw_dynamic_root_count": len(raw), "selected_root_count": len(selected),
            "arrays": {"path": "arrays.npz", "sha256": _sha256(arrays_path)},
        }
        _write_json(destination / "numerical_run.json", provenance)
    return raw, selected, provenance

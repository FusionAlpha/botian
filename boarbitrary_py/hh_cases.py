from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .case_registry import get_case_record
from .config import CaseInputs, CasePaths, SolverContext
from .initialize import initialize_case
from .io import load_case_inputs


class SourceFamilyMismatchError(ValueError):
    """Raised when an input capsule is handed to the wrong solver family."""


_CAPSULE_NAMES = {
    "bo2025_firehose_astfalk17_theta45": "firehose_astfalk17_theta45",
    "bo2025_ice_irvine18": "ice_irvine18_ice",
    "bo2025_ringbeam_umeda12_theta40": "ringbeam_umeda12_theta40",
    "bo2025_shell_min15_theta89": "shell_min15_theta89",
}

_SETUP_OVERRIDES: dict[str, dict[str, float | int]] = {
    "bo2025_ice_irvine18": {
        "n_harmonics": 22,
        "j_poles": 4,
        "pa1": 0.5,
        "pa2": 20.0,
        "dpa": 0.025,
        "theta_deg": 89.5,
        "b0": 2.1,
    },
}


def default_hh_cases_root() -> Path:
    return Path(__file__).resolve().parents[1] / "paper_cases"


def hh_case_paths(
    case_id: str,
    *,
    cases_root: str | Path | None = None,
    capsule_dir: str | Path | None = None,
) -> CasePaths:
    record = get_case_record(case_id)
    if record.source_family != "2025_bo_arbitrary_hh":
        raise SourceFamilyMismatchError(
            f"HH loader cannot load source family {record.source_family!r} for {case_id}"
        )
    root = (Path(cases_root) if cases_root is not None else default_hh_cases_root()).resolve()
    case_dir = (
        Path(capsule_dir).resolve()
        if capsule_dir is not None
        else root / _CAPSULE_NAMES[case_id]
    )
    input_dir = case_dir / "inputs_raw"
    return CasePaths(
        root=root,
        bohh_root=case_dir,
        input_dir=input_dir,
        modules_dir=case_dir,
        genfv_dir=case_dir / "coefficients",
        output_dir=case_dir,
        bo_in=input_dir / "bo.in",
        bo_setup=input_dir / "bo_setup.m",
        bo_knwn=input_dir / "bo_knwn.m",
        reference_mat=None,
    )


def load_hh_case_inputs(
    case_id: str,
    *,
    cases_root: str | Path | None = None,
    capsule_dir: str | Path | None = None,
) -> CaseInputs:
    paths = hh_case_paths(case_id, cases_root=cases_root, capsule_dir=capsule_dir)
    _validate_hh_capsule(paths, case_id)
    inputs = load_case_inputs(paths)
    overrides = _SETUP_OVERRIDES.get(case_id, {})
    if not overrides:
        return inputs

    setup = inputs.setup
    par = np.array(setup.par, dtype=float, copy=True)
    if "theta_deg" in overrides:
        par[1] = float(overrides["theta_deg"])
    if "pa1" in overrides and setup.ipa == 1:
        par[0] = float(overrides["pa1"])
    par[2] = np.cos(np.deg2rad(par[1])) * par[0]
    par[3] = np.sin(np.deg2rad(par[1])) * par[0]
    setup = replace(
        setup,
        n_harmonics=int(overrides.get("n_harmonics", setup.n_harmonics)),
        j_poles=int(overrides.get("j_poles", setup.j_poles)),
        b0=float(overrides.get("b0", setup.b0)),
        pa1=float(overrides.get("pa1", setup.pa1)),
        pa2=float(overrides.get("pa2", setup.pa2)),
        dpa=float(overrides.get("dpa", setup.dpa)),
        par=par,
    )
    return replace(inputs, setup=setup)


def initialize_hh_case(
    case_id: str,
    *,
    cases_root: str | Path | None = None,
    capsule_dir: str | Path | None = None,
) -> SolverContext:
    record = get_case_record(case_id)
    ctx = initialize_case(
        load_hh_case_inputs(case_id, cases_root=cases_root, capsule_dir=capsule_dir)
    )
    scan_points = len(ctx.pa) * len(ctx.pb)
    if ctx.nn != record.matrix_size or ctx.nw != record.roots_per_scan:
        raise ValueError(
            f"HH registry dimension mismatch for {case_id}: "
            f"computed nn={ctx.nn}, nw={ctx.nw}; "
            f"registered nn={record.matrix_size}, nw={record.roots_per_scan}"
        )
    if scan_points != record.scan.points:
        raise ValueError(
            f"HH registry scan mismatch for {case_id}: "
            f"computed {scan_points}, registered {record.scan.points}"
        )
    return ctx


def _validate_hh_capsule(paths: CasePaths, case_id: str) -> None:
    required = (paths.bo_in, paths.bo_setup, paths.bo_knwn)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete HH case capsule {case_id}: {missing}")

    first_row = _first_data_row(paths.bo_in)
    column_count = len(first_row.split())
    if column_count != 8:
        family = "legacy 13-column BO/PDRK" if column_count == 13 else "unknown"
        raise SourceFamilyMismatchError(
            f"HH loader expected the 8-column arbitrary-distribution schema for {case_id}; "
            f"found {column_count} columns ({family}) in {paths.bo_in}"
        )

    manifest = paths.output_dir / "manifest.yml"
    if manifest.is_file():
        source_id = _manifest_case_id(manifest)
        expected_id = _CAPSULE_NAMES[case_id]
        if source_id is not None and source_id != expected_id:
            raise SourceFamilyMismatchError(
                f"HH capsule identity mismatch: expected {expected_id!r}, found {source_id!r}"
            )


def _first_data_row(path: Path) -> str:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or line_number == 0:
            continue
        return stripped
    raise ValueError(f"No species rows found in {path}")


def _manifest_case_id(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("case_id:"):
            return line.split(":", 1)[1].strip().strip('"\'')
    return None

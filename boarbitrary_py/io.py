from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from .config import (
    BOInput,
    CaseInputs,
    CasePaths,
    FVCoefficients,
    SetupConfig,
    SpeciesInput,
)


def default_case_paths(root: str | Path | None = None) -> CasePaths:
    workspace = Path(root) if root is not None else Path.cwd()
    archive_root = workspace / "boarbitrary-main"
    bohh_root = archive_root / "em3dHH" / "boHH"
    output_dir = bohh_root / "output" / "firehose_Astfalk17"
    return CasePaths(
        root=archive_root,
        bohh_root=bohh_root,
        input_dir=output_dir,
        modules_dir=bohh_root / "modules",
        genfv_dir=bohh_root / "genfv",
        output_dir=output_dir,
        bo_in=output_dir / "bo.in",
        bo_setup=output_dir / "bo_setup.m",
        bo_knwn=output_dir / "bo_knwn.m",
        reference_mat=output_dir
        / "bo_S=2_theta=45_J=24_N=2_iem=1_npa=46_npb=1_B0=0.1_vdsz01=0_all.mat",
    )


def read_bo_in(path: str | Path) -> BOInput:
    rows: list[SpeciesInput] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no == 0 or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Expected 8 columns in {path}, got {len(parts)}: {line!r}")
            rows.append(
                SpeciesInput(
                    charge_qe=float(parts[0]),
                    mass_munit=float(parts[1]),
                    density=float(parts[2]),
                    tz_ev=float(parts[3]),
                    tp_ev=float(parts[4]),
                    drift_z_c=float(parts[5]),
                    drift_r_c=float(parts[6]),
                    ifv=int(float(parts[7])),
                )
            )
    if not rows:
        raise ValueError(f"No species rows found in {path}")
    return BOInput(tuple(rows))


def parse_firehose_setup(path: str | Path) -> SetupConfig:
    text = _strip_matlab_comments(Path(path).read_text(encoding="utf-8"))
    constants = _extract_physical_constants(text)

    par = np.zeros(6, dtype=float)
    par[0] = _extract_assignment(text, "par\\(1\\)", default=0.5)
    par[1] = _extract_assignment(text, "par\\(2\\)", default=45.0)
    par[2] = np.cos(np.deg2rad(par[1])) * par[0]
    par[3] = np.sin(np.deg2rad(par[1])) * par[0]

    wg0 = _extract_complex_assignment(text, "wg0", default=6.31727e1 + 4.19534j)

    savepath_match = re.search(r"savepath\s*=\s*'([^']+)'", text)
    savepath = savepath_match.group(1) if savepath_match else "../output/firehose_Astfalk17/"

    return SetupConfig(
        savepath=savepath,
        iem=int(_extract_assignment(text, "iem", default=1)),
        n_harmonics=int(_extract_assignment(text, "N", default=2)),
        j_poles=int(_extract_assignment(text, "J", default=24)),
        sp=int(_extract_assignment(text, "sp", default=0)),
        wg0=wg0,
        b0=_extract_assignment(text, "B0", default=0.1),
        par=par,
        ipa=int(_extract_assignment(text, "ipa", default=1)),
        ipb=int(_extract_assignment(text, "ipb", default=1)),
        iloga=int(_extract_assignment(text, "iloga", default=0)),
        ilogb=int(_extract_assignment(text, "ilogb", default=0)),
        pa1=_extract_assignment(text, "pa1", default=0.05),
        pa2=_extract_assignment(text, "pa2", default=0.5),
        dpa=_extract_assignment(text, "dpa", default=0.01),
        pb1=_extract_assignment(text, "pb1", default=0.0),
        pb2=_extract_assignment(text, "pb2", default=89.0),
        dpb=_extract_assignment(text, "dpb", default=2.5),
        iout=int(_extract_assignment(text, "iout", default=0)),
        c2=constants["c2"],
        epsilon0=constants["epsilon0"],
        mu0=constants["mu0"],
        kb=constants["kb"],
        qe=constants["qe"],
        mp=constants["mp"],
        me=constants["me"],
        munit=constants["munit"],
    )


def load_fv_coefficients(path: str | Path) -> FVCoefficients:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    fvc = data["fvc"]
    return FVCoefficients(
        alm=np.asarray(fvc.alm, dtype=np.complex128),
        vdz=float(fvc.vdz),
        vdr=float(fvc.vdr),
        vtz=float(fvc.vtz),
        vtp=float(fvc.vtp),
    )


def load_case_inputs(paths: CasePaths | None = None) -> CaseInputs:
    paths = paths or default_case_paths()
    bo_input = read_bo_in(paths.bo_in)
    setup = parse_firehose_setup(paths.bo_setup)
    fv_coefficients: dict[int, FVCoefficients] = {}
    for index, species in enumerate(bo_input.species, start=1):
        if species.ifv == 0:
            fv_coefficients[index] = load_fv_coefficients(paths.genfv_dir / f"fvceff{index}.mat")
    return CaseInputs(paths, bo_input, setup, fv_coefficients)


def load_reference_output(path: str | Path):
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def save_scan_result(
    path: str | Path,
    *,
    ww: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
    pa_used: np.ndarray | None = None,
    pb_used: np.ndarray | None = None,
    pa_full: np.ndarray | None = None,
    pb_full: np.ndarray | None = None,
    scan_indices: np.ndarray | None = None,
) -> None:
    payload = {"ww": ww, "pa": pa, "pb": pb}
    optional = {
        "pa_used": pa_used,
        "pb_used": pb_used,
        "pa_full": pa_full,
        "pb_full": pb_full,
        "scan_indices": scan_indices,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    savemat(path, payload)


def _strip_matlab_comments(text: str) -> str:
    stripped_lines = []
    for line in text.splitlines():
        stripped_lines.append(line.split("%", 1)[0])
    return "\n".join(stripped_lines)


def _extract_assignment(text: str, name_pattern: str, *, default: float) -> float:
    pattern = rf"(?<![A-Za-z0-9_]){name_pattern}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*;"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else default


def _extract_complex_assignment(text: str, name: str, *, default: complex) -> complex:
    pattern = rf"(?<![A-Za-z0-9_]){name}\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*([-+])\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*\*\s*1i\s*;"
    match = re.search(pattern, text)
    if not match:
        return default
    real = float(match.group(1))
    imag = float(match.group(3))
    if match.group(2) == "-":
        imag = -imag
    return real + 1j * imag


def _extract_physical_constants(text: str) -> dict[str, float]:
    c2_default = (2.99792458e8) ** 2
    constants = {
        "c2": c2_default,
        "epsilon0": 8.854187817e-12,
        "mu0": 1.0 / (c2_default * 8.854187817e-12),
        "kb": 1.38064852e-23,
        "qe": 1.60217662e-19,
        "mp": 1.6726219e-27,
        "me": 9.1094e-31,
    }
    c2_expr = _last_assignment_expression(text, "c2")
    if c2_expr is not None:
        constants["c2"] = _eval_constant_expression(c2_expr, constants)
    for source_name, key in (("epsilon0", "epsilon0"), ("mu0", "mu0"), ("kB", "kb"), ("qe", "qe"), ("mp", "mp"), ("me", "me")):
        expr = _last_assignment_expression(text, source_name)
        if expr is not None:
            constants[key] = _eval_constant_expression(expr, constants)
    munit_expr = _last_assignment_expression(text, "munit")
    if munit_expr is None:
        constants["munit"] = constants["mp"]
    else:
        constants["munit"] = _eval_constant_expression(munit_expr, constants)
    return constants


def _last_assignment_expression(text: str, name: str) -> str | None:
    matches = re.findall(rf"(?<![A-Za-z0-9_]){name}\s*=\s*([^;]+);", text)
    return matches[-1].strip() if matches else None


def _eval_constant_expression(expr: str, variables: dict[str, float]) -> float:
    compact = re.sub(r"\s+", "", expr)
    number = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
    if re.fullmatch(number, compact):
        return float(compact)
    squared = re.fullmatch(rf"\(({number})\)\^2", compact)
    if squared:
        return float(squared.group(1)) ** 2
    if compact == "1/(c2*mu0)":
        return 1.0 / (variables["c2"] * variables["mu0"])
    if compact == "1/(c2*epsilon0)":
        return 1.0 / (variables["c2"] * variables["epsilon0"])
    if compact in variables:
        return float(variables[compact])
    raise NotImplementedError(f"Unsupported physical constant expression: {expr}")

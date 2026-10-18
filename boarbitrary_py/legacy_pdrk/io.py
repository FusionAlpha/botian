from __future__ import annotations

import json
from pathlib import Path

from .config import (
    LegacyCanonicalConfig,
    LegacyCaseConfig,
    LegacyCaseInputs,
    LegacyModelConfig,
    LegacyScanConfig,
    LegacySpeciesInput,
)


LEGACY_CASE_IDS = (
    "bo2018_beammode",
    "bo2018_coldplasma",
    "bo2018_darwin",
    "bo2018_dispersionsurface",
    "bo2018_es1d_beam",
    "bo2018_es3d_losscone",
    "bo2018_loss_cone_mirror",
    "bo2018_muschietti17",
    "bo2018_ringbeam_umeda12",
)


def default_legacy_cases_root() -> Path:
    return Path(__file__).resolve().parents[2] / "legacy_cases"


def read_legacy_bo_in(path: str | Path) -> tuple[LegacySpeciesInput, ...]:
    rows: list[LegacySpeciesInput] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no == 0 or not line.strip() or line.lstrip().startswith("%"):
                continue
            parts = line.split()
            if len(parts) != 13:
                raise ValueError(f"Expected 13 legacy BO columns in {path}, got {len(parts)}: {line!r}")
            values = [float(part) for part in parts]
            rows.append(
                LegacySpeciesInput(
                    charge_qe=values[0],
                    mass_munit=values[1],
                    density=values[2],
                    tz_ev=values[3],
                    tp_ev=values[4],
                    alpha=values[5],
                    delta=values[6],
                    drift_z_c=values[7],
                    drift_x_c=values[8],
                    drift_y_c=values[9],
                    drift_r_c=values[10],
                    collision_frequency=values[11],
                    magnetized=bool(int(values[12])),
                )
            )
    if not rows:
        raise ValueError(f"No legacy species rows found in {path}")
    return tuple(rows)


def load_legacy_case_config(
    case_id: str,
    *,
    cases_root: str | Path | None = None,
) -> LegacyCaseConfig:
    if case_id not in LEGACY_CASE_IDS:
        raise KeyError(f"Unknown legacy case {case_id!r}; choices: {LEGACY_CASE_IDS}")
    root = (Path(cases_root) if cases_root is not None else default_legacy_cases_root()).resolve()
    case_root = root / case_id
    data = json.loads((case_root / "case.json").read_text(encoding="utf-8"))
    if data["case_id"] != case_id or data["source_family"] != "2018_bo_pdrk":
        raise ValueError(f"Legacy source-family mismatch in {case_root / 'case.json'}")

    model = data["model"]
    scan = data["scan"]
    canonical = data["canonical"]
    raw_dir = case_root / data["raw_input_dir"]
    required = [raw_dir / "bo.in", raw_dir / "bo_setup.m", raw_dir / "bo_knwn.m"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete legacy case capsule {case_id}: {missing}")
    return LegacyCaseConfig(
        schema_version=int(data["schema_version"]),
        case_id=case_id,
        source_family=data["source_family"],
        source_example=data["source_example"],
        root=case_root,
        raw_input_dir=raw_dir,
        model=LegacyModelConfig(
            iem=int(model["iem"]),
            n_harmonics=int(model["n_harmonics"]),
            j_poles=int(model["j_poles"]),
            jpole_variant=str(model["jpole_variant"]),
            sp=int(model["sp"]),
            set_ns=tuple((int(row[0]), int(row[1])) for row in model["set_ns"]),
            set_ns_semantics=str(model["set_ns_semantics"]),
        ),
        b0=float(data["physics"]["b0"]),
        constant_system=str(data["physics"]["constant_system"]),
        scan=LegacyScanConfig(**{key: float(value) if key not in {"ipa", "ipb", "iloga", "ilogb"} else int(value) for key, value in scan.items()}),
        normalization_recipe=str(data["normalization"]["recipe"]),
        canonical=LegacyCanonicalConfig(
            scan_points=int(canonical["scan_points"]),
            coordinate_dimensions=int(canonical["coordinate_dimensions"]),
            matrix_size=int(canonical["matrix_size"]),
            roots_per_scan=int(canonical["roots_per_scan"]),
            overrides=dict(canonical["overrides"]),
        ),
        development_oracle=str(data["development_oracle"]),
    )


def load_legacy_case(
    case_id: str,
    *,
    cases_root: str | Path | None = None,
) -> LegacyCaseInputs:
    config = load_legacy_case_config(case_id, cases_root=cases_root)
    species = read_legacy_bo_in(config.raw_input_dir / "bo.in")
    return LegacyCaseInputs(config=config, species=species)

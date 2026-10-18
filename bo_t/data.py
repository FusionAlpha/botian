"""Verified access to the compact frozen prediction arrays (redraw mode only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_frozen(case_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    directory = ROOT / "data" / "frozen"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["cases"][case_name]
    path = directory / record["file"]
    if path.parent.resolve() != directory.resolve():
        raise ValueError("Frozen data path must stay inside data/frozen")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"Frozen data checksum mismatch: {case_name}")
    with np.load(path, allow_pickle=False) as arrays:
        raw = arrays["raw"].copy()
        selected = arrays["selected"].copy()
    for values, count_key in ((raw, "raw_count"), (selected, "selected_count")):
        if values.shape != (record[count_key], 3) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid frozen array shape or nonfinite values: {case_name}")
    if len(np.unique(raw[:, 0])) != record["expected_scans"]:
        raise ValueError(f"Frozen scan coverage mismatch: {case_name}")
    return raw, selected, {
        "mode": "cached", "numerical_recomputation": False,
        "data_file": path.relative_to(ROOT).as_posix(), "data_sha256": digest,
        "scan_count": record["expected_scans"],
        "original_verified_provenance": record["original_verified_provenance"],
    }

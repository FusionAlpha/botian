from __future__ import annotations

"""Crash-safe persistence for immutable BO selector predictions.

The checksum file is written last and therefore acts as the commit marker for
the JSON/CSV pair.  Readers never accept a prediction without verifying both
files and the schema-level canonical manifest hash.
"""

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .selector_models import PredictionManifest


@dataclass(frozen=True)
class FrozenPredictionPaths:
    json_path: Path
    csv_path: Path
    checksum_path: Path


@dataclass(frozen=True)
class VerifiedPredictionArtifact:
    manifest: PredictionManifest
    paths: FrozenPredictionPaths
    json_sha256: str
    csv_sha256: str


@dataclass(frozen=True)
class FrozenDiagnosticPaths:
    json_path: Path
    checksum_path: Path


def freeze_prediction_artifact(
    manifest: PredictionManifest,
    output_dir: str | Path,
    *,
    stem: str | None = None,
) -> FrozenPredictionPaths:
    """Atomically commit prediction JSON, audit CSV, and SHA-256 metadata.

    Each payload is written and fsynced through a same-directory temporary
    file.  The checksum sidecar is replaced last, so a reader observes either
    no committed artifact or a pair whose bytes can be verified.
    """

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifact_stem = stem or (
        f"{manifest.source.case_id}__scan-{manifest.source.scan_index:04d}__prediction"
    )
    paths = FrozenPredictionPaths(
        json_path=destination / f"{artifact_stem}.json",
        csv_path=destination / f"{artifact_stem}.csv",
        checksum_path=destination / f"{artifact_stem}.sha256",
    )

    json_bytes = (
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    csv_bytes = _prediction_csv_bytes(manifest)
    json_hash = _sha256_bytes(json_bytes)
    csv_hash = _sha256_bytes(csv_bytes)
    checksum_bytes = (
        f"{json_hash}  {paths.json_path.name}\n"
        f"{csv_hash}  {paths.csv_path.name}\n"
        f"manifest_content_sha256  {manifest.content_sha256()}\n"
    ).encode("ascii")

    staged: list[tuple[Path, Path]] = []
    try:
        staged.append((_stage_bytes(paths.json_path, json_bytes), paths.json_path))
        staged.append((_stage_bytes(paths.csv_path, csv_bytes), paths.csv_path))
        staged.append((_stage_bytes(paths.checksum_path, checksum_bytes), paths.checksum_path))
        os.replace(staged[0][0], paths.json_path)
        os.replace(staged[1][0], paths.csv_path)
        os.replace(staged[2][0], paths.checksum_path)
        _fsync_directory(destination)
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    return paths


def verify_prediction_artifact(
    prediction_json_path: str | Path,
) -> VerifiedPredictionArtifact:
    """Verify the committed bytes and deserialize the immutable manifest."""

    json_path = Path(prediction_json_path).resolve()
    checksum_path = json_path.with_suffix(".sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(f"prediction commit checksum is missing: {checksum_path}")

    entries, manifest_content_hash = _read_checksum_file(checksum_path)
    expected_json_name = json_path.name
    if expected_json_name not in entries:
        raise ValueError("prediction checksum does not name the requested JSON file")
    csv_names = [name for name in entries if name.lower().endswith(".csv")]
    if len(csv_names) != 1:
        raise ValueError("prediction checksum must name exactly one CSV evidence file")
    csv_name = csv_names[0]
    if Path(csv_name).name != csv_name:
        raise ValueError("prediction checksum contains an unsafe CSV path")
    csv_path = json_path.parent / csv_name
    if not json_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("committed prediction JSON/CSV pair is incomplete")

    actual_json_hash = _sha256_path(json_path)
    actual_csv_hash = _sha256_path(csv_path)
    if actual_json_hash != entries[expected_json_name]:
        raise ValueError("frozen prediction JSON checksum mismatch")
    if actual_csv_hash != entries[csv_name]:
        raise ValueError("frozen prediction CSV checksum mismatch")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    manifest = PredictionManifest.from_dict(payload)
    if manifest.content_sha256() != manifest_content_hash:
        raise ValueError("frozen prediction canonical manifest hash mismatch")
    if _csv_raw_indices(csv_path) != tuple(sorted(item.raw_index for item in manifest.roots)):
        raise ValueError("frozen prediction CSV root rows differ from the JSON manifest")
    return VerifiedPredictionArtifact(
        manifest=manifest,
        paths=FrozenPredictionPaths(json_path, csv_path, checksum_path),
        json_sha256=actual_json_hash,
        csv_sha256=actual_csv_hash,
    )


def freeze_selection_diagnostic(
    payload: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str,
) -> FrozenDiagnosticPaths:
    """Commit a machine-readable whole-scan abstention record."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{stem}.diagnostic.json"
    checksum_path = destination / f"{stem}.diagnostic.sha256"
    json_bytes = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    checksum_bytes = f"{_sha256_bytes(json_bytes)}  {json_path.name}\n".encode("ascii")
    staged = [
        (_stage_bytes(json_path, json_bytes), json_path),
        (_stage_bytes(checksum_path, checksum_bytes), checksum_path),
    ]
    try:
        os.replace(staged[0][0], json_path)
        os.replace(staged[1][0], checksum_path)
        _fsync_directory(destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return FrozenDiagnosticPaths(json_path=json_path, checksum_path=checksum_path)


def _prediction_csv_bytes(manifest: PredictionManifest) -> bytes:
    fields = (
        "raw_index_0based",
        "retained_position_0based",
        "root_real",
        "root_imag",
        "final_state",
        "corrected_center_real",
        "corrected_center_imag",
        "isolation_radius",
        "stability_tolerance",
        "declared_pole_real",
        "declared_pole_imag",
        "pole_multiplicity",
        "pole_multiplicity_slot_0based",
        "pole_matching_distance",
        "pole_provenance",
        "stage_statuses_json",
        "probe_evidence_json",
        "failed_probe_json",
        "rigorous",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for item in sorted(manifest.roots, key=lambda value: value.raw_index):
        pole_evidence = item.structural_pole
        pole = None if pole_evidence is None else pole_evidence.declared_pole
        center = item.corrected_center
        writer.writerow(
            {
                "raw_index_0based": item.raw_index,
                "retained_position_0based": (
                    "" if item.retained_position is None else item.retained_position
                ),
                "root_real": float(item.source_root.real),
                "root_imag": float(item.source_root.imag),
                "final_state": item.final_state.value,
                "corrected_center_real": "" if center is None else float(center.real),
                "corrected_center_imag": "" if center is None else float(center.imag),
                "isolation_radius": (
                    "" if item.isolation_radius is None else item.isolation_radius
                ),
                "stability_tolerance": item.stability_tolerance,
                "declared_pole_real": "" if pole is None else float(pole.value_normalized.real),
                "declared_pole_imag": "" if pole is None else float(pole.value_normalized.imag),
                "pole_multiplicity": "" if pole is None else pole.multiplicity,
                "pole_multiplicity_slot_0based": (
                    "" if pole_evidence is None else pole_evidence.multiplicity_slot
                ),
                "pole_matching_distance": (
                    "" if pole_evidence is None else pole_evidence.matching_distance
                ),
                "pole_provenance": "" if pole is None else pole.provenance,
                "stage_statuses_json": _compact_json(dict(item.stage_statuses)),
                "probe_evidence_json": _compact_json(
                    [probe.to_dict() for probe in item.probes]
                ),
                "failed_probe_json": (
                    "" if item.failed_probe is None else _compact_json(item.failed_probe.to_dict())
                ),
                "rigorous": "false",
            }
        )
    return stream.getvalue().encode("utf-8")


def _read_checksum_file(path: Path) -> tuple[dict[str, str], str]:
    entries: dict[str, str] = {}
    manifest_content_hash: str | None = None
    for raw_line in path.read_text(encoding="ascii").splitlines():
        parts = raw_line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid prediction checksum line: {raw_line!r}")
        left, right = parts
        if left == "manifest_content_sha256":
            _require_sha256(right)
            manifest_content_hash = right
            continue
        _require_sha256(left)
        if Path(right).name != right or right in entries:
            raise ValueError("prediction checksum contains duplicate or unsafe filenames")
        entries[right] = left
    if manifest_content_hash is None:
        raise ValueError("prediction checksum omits manifest_content_sha256")
    return entries, manifest_content_hash


def _csv_raw_indices(path: Path) -> tuple[int, ...]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return tuple(int(row["raw_index_0based"]) for row in rows)


def _stage_bytes(destination: Path, payload: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"invalid SHA-256 value: {value!r}")


__all__ = [
    "FrozenDiagnosticPaths",
    "FrozenPredictionPaths",
    "VerifiedPredictionArtifact",
    "freeze_prediction_artifact",
    "freeze_selection_diagnostic",
    "verify_prediction_artifact",
]

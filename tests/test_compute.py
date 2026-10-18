"""Fast contract tests; numerical smoke runs are exercised by the release CLI."""

from pathlib import Path

import numpy as np
import pytest

from bo_t import compute


def test_recipes_preserve_all_1061_points_and_selector_settings():
    assert sum(leg.scan_count for recipe in compute.RECIPES.values() for leg in recipe.legs) == 1061
    assert compute.RECIPES["signed_em"].audit_all_roots
    assert compute.RECIPES["loss_cone_mirror"].legs[0].x_coordinate == "kz_normalized"
    assert compute.RECIPES["es1d_beam"].quadrature_order == 120
    assert compute.RECIPES["firehose"].quadrature_order == 80


def test_hh_profile_retains_forward_j24_development_qualification():
    leg = compute.RECIPES["hh_ring_beam"].legs[0]
    profile = compute._profile("hh_ring_beam", leg)
    assert profile.source.discretization.pole_order == 12
    assert profile.source.discretization.harmonic_truncation == 6
    assert profile.fallback.discretization.pole_order == 24
    assert not profile.scan_scope.held_out
    assert dict(profile.metadata)["evaluation_labels_read"] is True
    assert dict(profile.metadata)["declared_before_evaluation"] is False


def test_checked_json_detects_corruption(tmp_path):
    path = tmp_path / "commit.json"
    compute._write_json(path, {"count": 3})
    assert compute._read_json(path) == {"count": 3}
    path.write_text('{"count": 4}', encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum"):
        compute._read_json(path)


def test_unknown_case_and_existing_output_fail_before_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "run_registered_case", lambda *args, **kwargs: pytest.fail("solver called"))
    with pytest.raises(ValueError, match="Unknown case"):
        compute.recompute("unknown", tmp_path)
    (tmp_path / "old.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        compute.recompute("es1d_beam", tmp_path)
    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "keep"


def test_smoke_scope_resume_and_stale_resume(tmp_path, monkeypatch):
    """Stub only expensive scans; exercise real plan, resume and output logic."""
    inputs = {"inputs/example/case.json": "a" * 64}
    code = {"bo_t/compute.py": "b" * 64}
    monkeypatch.setattr(compute, "_input_hashes", lambda recipe: dict(inputs))
    monkeypatch.setattr(compute, "_code_hashes", lambda: dict(code))
    monkeypatch.setattr(compute, "_environment", lambda: {"test_environment": True})
    called = []

    def fake_compute(case_name, recipe, leg, index, directory, fingerprint):
        called.append((case_name, leg.name, index))
        compute._write_json(directory / "scan_commit.json", {
            "source_seconds": 0.1, "scan_wall_seconds": 0.2,
        })

    def fake_read(directory, leg, index, fingerprint):
        commit = compute._read_json(directory / "scan_commit.json")
        values = np.asarray([[float(index), 1.0, 0.1]])
        return values, values, commit

    monkeypatch.setattr(compute, "_compute_scan", fake_compute)
    monkeypatch.setattr(compute, "_read_scan", fake_read)
    output = tmp_path / "result"
    raw, selected, provenance = compute.recompute("signed_em", output, smoke=True)
    assert called == [("signed_em", "negative", 125), ("signed_em", "positive", 125)]
    assert raw.shape == selected.shape == (2, 3)
    assert not provenance["full_scan"]
    assert provenance["actual_scan_count"] == 2
    assert provenance["registered_scan_count"] == 500
    assert not provenance["frozen_plotting_data_read"]
    assert all(not Path(item["path"]).is_absolute() for item in provenance["scans"])
    _, _, resumed = compute.recompute("signed_em", output, smoke=True, resume=True)
    assert len(called) == 2
    assert resumed["resumed_scan_count"] == 2
    inputs["inputs/example/case.json"] = "c" * 64
    with pytest.raises(ValueError, match="Resume rejected"):
        compute.recompute("signed_em", output, smoke=True, resume=True)


def test_solver_failure_never_falls_back_to_cached_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "_input_hashes", lambda recipe: {})
    monkeypatch.setattr(compute, "_code_hashes", lambda: {})
    monkeypatch.setattr(compute, "_environment", lambda: {})

    def fail(*args, **kwargs):
        raise RuntimeError("intentional source failure")

    monkeypatch.setattr(compute, "run_registered_case", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional source failure"):
        compute.recompute("es1d_beam", output, smoke=True)
    assert not (output / "arrays.npz").exists()
    assert not (output / "numerical_run.json").exists()


def test_output_cannot_modify_bundled_inputs():
    with pytest.raises(ValueError, match="bundled source or data"):
        compute.recompute("es1d_beam", compute.ROOT / "inputs" / "bad")

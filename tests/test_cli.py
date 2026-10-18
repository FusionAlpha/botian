"""CLI contracts with expensive calculation and rendering replaced by stubs."""

import json
import os

import numpy as np
import pytest

from bo_t import cli, compute, data, plotting


@pytest.fixture
def routed_cli(tmp_path, monkeypatch):
    root = tmp_path / "release"
    monkeypatch.setattr(cli, "ROOT", root)
    for name in (
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        "BO_DENSE_EIG_THREADS", "MPLBACKEND",
    ):
        monkeypatch.setenv(name, os.environ.get(name, "1"))
    calls = []
    values = np.asarray([[0.2, 1.0, -0.1]])

    def fresh(name, output, *, smoke, resume):
        calls.append(("compute", name, output, smoke, resume))
        return values.copy(), values.copy(), {
            "numerical_recomputation": True, "full_scan": not smoke,
        }

    def frozen(name):
        calls.append(("cached", name))
        return values.copy(), values.copy(), {"numerical_recomputation": False}

    def render(name, raw, selected, output):
        calls.append(("render", name, output))
        np.testing.assert_array_equal(raw, values)
        np.testing.assert_array_equal(selected, values)
        return {"case_name": name, "selected_root_count": len(selected)}

    monkeypatch.setattr(compute, "recompute", fresh)
    monkeypatch.setattr(data, "load_frozen", frozen)
    monkeypatch.setattr(plotting, "render_case", render)
    return root, calls


def test_default_mode_recomputes_all_cases_and_never_reads_cache(routed_cli):
    root, calls = routed_cli
    assert cli.main([]) == 0
    names = [case.name for case in plotting.CASES]
    assert [call[1] for call in calls if call[0] == "compute"] == names
    assert [call[1] for call in calls if call[0] == "render"] == names
    assert not any(call[0] == "cached" for call in calls)
    assert all(call[3:] == (False, False) for call in calls if call[0] == "compute")
    manifest = json.loads((root / "results/recompute/manifest_all.json").read_text())
    assert manifest["mode"] == "recompute"
    assert manifest["numerical_recomputation"] is True
    assert manifest["full_scan_requested"] is True
    assert set(manifest["cases"]) == set(names)


def test_cached_mode_never_enters_numeric_path(routed_cli):
    root, calls = routed_cli
    assert cli.main(["--mode", "cached", "--case", "firehose"]) == 0
    assert [call[0] for call in calls] == ["cached", "render"]
    manifest = json.loads((root / "results/cached/manifest_firehose.json").read_text())
    assert manifest["mode"] == "cached"
    assert manifest["numerical_recomputation"] is False
    assert manifest["cases"]["firehose"]["provenance"]["numerical_recomputation"] is False


def test_smoke_and_resume_are_forwarded_explicitly(routed_cli, tmp_path):
    _, calls = routed_cli
    output = tmp_path / "custom"
    assert cli.main(["--mode", "smoke", "--case", "signed_em",
                     "--resume", "--output", str(output)]) == 0
    assert calls[0] == ("compute", "signed_em", output / "signed_em", True, True)
    assert calls[1] == ("render", "signed_em", output / "signed_em/figures")
    manifest = json.loads((output / "manifest_signed_em.json").read_text())
    assert manifest["mode"] == "smoke"
    assert manifest["numerical_recomputation"] is True
    assert manifest["full_scan_requested"] is False


def test_solver_failure_is_not_hidden_by_cached_redraw(routed_cli, monkeypatch):
    root, calls = routed_cli

    def fail(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(compute, "recompute", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        cli.main(["--case", "es1d_beam"])
    assert calls == []
    assert not (root / "results/recompute/manifest_es1d_beam.json").exists()


def test_corrupt_frozen_data_are_not_silently_recomputed(routed_cli, monkeypatch):
    root, calls = routed_cli

    def corrupt(*args, **kwargs):
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(data, "load_frozen", corrupt)
    with pytest.raises(ValueError, match="checksum mismatch"):
        cli.main(["--mode", "cached", "--case", "es1d_beam"])
    assert calls == []
    assert not (root / "results/cached/manifest_es1d_beam.json").exists()


@pytest.mark.parametrize("protected", ["inputs", "data/frozen", "bo_t", "boarbitrary_py"])
def test_source_and_data_output_paths_are_rejected(routed_cli, protected):
    root, calls = routed_cli
    target = root / protected / "new-output"
    with pytest.raises(SystemExit) as error:
        cli.main(["--mode", "cached", "--case", "es1d_beam", "--output", str(target)])
    assert error.value.code == 2
    assert calls == []
    assert not target.exists()


def test_duplicate_case_rejected_before_work(routed_cli):
    _, calls = routed_cli
    with pytest.raises(SystemExit) as error:
        cli.main(["--case", "firehose", "firehose"])
    assert error.value.code == 2
    assert calls == []


def test_unknown_case_rejected_before_work(routed_cli):
    _, calls = routed_cli
    with pytest.raises(SystemExit) as error:
        cli.main(["--case", "unknown"])
    assert error.value.code == 2
    assert calls == []


@pytest.mark.parametrize("previous,requested", [
    ("recompute", "cached"), ("cached", "recompute"),
    ("smoke", "recompute"), ("recompute", "smoke"),
    ("cached", "smoke"), ("smoke", "cached"),
])
def test_output_mode_cannot_be_changed(routed_cli, tmp_path, previous, requested):
    _, calls = routed_cli
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "run_mode.json"
    original = json.dumps({"mode": previous})
    marker.write_text(original, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["--mode", requested, "--case", "es1d_beam", "--output", str(output)])
    assert error.value.code == 2
    assert calls == []
    assert marker.read_text(encoding="utf-8") == original
    assert sorted(path.name for path in output.iterdir()) == ["run_mode.json"]


@pytest.mark.parametrize("previous,requested", [
    ("recompute", "cached"), ("cached", "recompute"), ("smoke", "cached"),
])
def test_old_manifest_blocks_mixed_mode_even_without_marker(
        routed_cli, tmp_path, previous, requested):
    _, calls = routed_cli
    output = tmp_path / "legacy-output"
    output.mkdir()
    manifest = output / "manifest_all.json"
    original = json.dumps({"mode": previous, "cases": {}})
    manifest.write_text(original, encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["--mode", requested, "--case", "firehose", "--output", str(output)])
    assert error.value.code == 2
    assert calls == []
    assert manifest.read_text(encoding="utf-8") == original
    assert not (output / "run_mode.json").exists()


def test_cached_mode_cannot_overwrite_incomplete_numerical_run(routed_cli, tmp_path):
    _, calls = routed_cli
    output = tmp_path / "interrupted"
    case = output / "es1d_beam"
    case.mkdir(parents=True)
    plan = case / "run_plan.json"
    plan.write_text('{"mode": "recompute"}', encoding="utf-8")
    figure = case / "figures" / "existing.svg"
    figure.parent.mkdir()
    figure.write_text("keep existing figure", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["--mode", "cached", "--case", "es1d_beam", "--output", str(output)])
    assert error.value.code == 2
    assert calls == []
    assert figure.read_text(encoding="utf-8") == "keep existing figure"
    assert plan.read_text(encoding="utf-8") == '{"mode": "recompute"}'
    assert not (output / "run_mode.json").exists()


def test_cached_redraw_can_repeat_within_its_own_output(routed_cli):
    root, calls = routed_cli
    arguments = ["--mode", "cached", "--case", "firehose"]
    assert cli.main(arguments) == cli.main(arguments) == 0
    assert [call[0] for call in calls] == ["cached", "render", "cached", "render"]
    marker = json.loads((root / "results/cached/run_mode.json").read_text())
    assert marker["mode"] == "cached"

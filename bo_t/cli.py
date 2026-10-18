"""One entry point for fresh calculations, smoke tests, and cached redraws."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("recompute", "cached", "smoke"), default="recompute",
                        help="Default: solve all scan points and select roots anew. Cached mode only redraws saved predictions.")
    parser.add_argument("--case", nargs="+", choices=(
        "es1d_beam", "umeda_ring_beam", "signed_em", "hh_ring_beam",
        "firehose", "muschietti_beam", "loss_cone_mirror"), help="Default: all seven cases")
    parser.add_argument("--output", type=Path, help="Default: results/<mode> inside this source tree")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse committed scans from an interrupted run")
    args = parser.parse_args(argv)
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "BO_DENSE_EIG_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    from threadpoolctl import threadpool_limits
    from .plotting import CASES, render_case

    names = args.case or [case.name for case in CASES]
    if len(set(names)) != len(names):
        parser.error("Each case may be selected only once")
    output = (args.output or ROOT / "results" / args.mode).resolve()
    # Do not allow a convenience output option to overwrite release inputs.
    for protected in (ROOT / "inputs", ROOT / "data", ROOT / "bo_t", ROOT / "boarbitrary_py"):
        if output == protected or protected in output.parents:
            parser.error("--output must not be inside source code or bundled input directories")
    mode_marker = output / "run_mode.json"
    if mode_marker.exists():
        recorded_mode = json.loads(mode_marker.read_text(encoding="utf-8"))["mode"]
        if recorded_mode != args.mode:
            parser.error(f"Output already belongs to mode {recorded_mode!r}; choose another --output")
    for existing in output.glob("manifest_*.json"):
        recorded_mode = json.loads(existing.read_text(encoding="utf-8")).get("mode")
        if recorded_mode != args.mode:
            parser.error("Output contains results from another mode; choose another --output")
    if args.mode == "cached" and any((output / name / "run_plan.json").exists() for name in names):
        parser.error("Cached redraw cannot overwrite figures from a numerical run")
    output.mkdir(parents=True, exist_ok=True)
    mode_marker.write_text(json.dumps({"mode": args.mode}, indent=2) + "\n", encoding="utf-8")
    results = {}
    with threadpool_limits(limits=1, user_api="blas"):
        for name in names:
            print(f"[{args.mode}] {name}", flush=True)
            if args.mode == "cached":
                from .data import load_frozen
                raw, selected, provenance = load_frozen(name)
            else:
                from .compute import recompute
                raw, selected, provenance = recompute(
                    name, output / name, smoke=args.mode == "smoke", resume=args.resume)
            figure = render_case(name, raw, selected, output / name / "figures")
            results[name] = {"provenance": provenance, "figure": figure}
            print(f"  wrote {name}/figures", flush=True)
    manifest = {
        "schema_version": 1, "mode": args.mode,
        "numerical_recomputation": args.mode != "cached",
        "full_scan_requested": args.mode != "smoke", "blas_threads": 1,
        "cases": results,
    }
    # Per-case manifests survive interrupted multi-case invocations.
    label = "all" if len(names) == len(CASES) else "_".join(names)
    manifest_path = output / f"manifest_{label}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Completed {len(results)} case(s). Manifest: {manifest_path}", flush=True)
    return 0

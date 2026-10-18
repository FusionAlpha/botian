# BO-T

BO-T (BO-Tian) is a Python implementation of response-preserving matrix reduction and root selection for BO/PDRK kinetic plasma dispersion calculations. This source tree contains the numerical inputs and plotting pipeline for the seven active manuscript examples.

## Reference

[Tian2026] Xizhi Tian, Huasheng Xie and Long Wei, Accelerating Computation and Resolving Extraneous Roots in the Kinetic Plasma Dispersion Relation Solver BO, to be submitted, 2026.

## Reproduce the figures

Install Python 3.11 or newer, then run from this directory:

```powershell
# Windows PowerShell
.\run.ps1
```

```sh
# Linux or macOS
sh run.sh
```

The launchers create a local `.venv`, install the reference dependencies from `requirements.txt` and recompute all seven examples. No MATLAB, MATLAB Runtime, GPU or remote server is required. Initial installation needs access to the Python package index; subsequent numerical calculations use bundled inputs. All numerical runs use one BLAS thread.

Full recomputation includes 1,061 scan points and the root-selection stage. It is a research calculation, not a quick plotting demo. To check the installation first:

```sh
python -m pip install -r requirements.txt -e .
python reproduce.py --mode smoke --case es1d_beam
```

To redraw the manuscript figure data without rerunning the solver:

```sh
python reproduce.py --mode cached
```

The modes are explicit:

| Mode | What runs | Default output |
| --- | --- | --- |
| `recompute` (default) | Source eigensolves, candidate selection and plotting over the complete requested scans | `results/recompute/` |
| `smoke` | One midpoint per requested example (both signs for signed EM), followed by plotting | `results/smoke/` |
| `cached` | Checksum-verified redraw of bundled frozen numerical arrays; no eigensolve or selector | `results/cached/` |

Run a single example or choose an output directory with:

```sh
python reproduce.py --case firehose --output results/firehose
python reproduce.py --help
```

Available cases are `es1d_beam`, `umeda_ring_beam`, `signed_em`, `hh_ring_beam`, `firehose`, `muschietti_beam` and `loss_cone_mirror`. Omitting `--case` runs all seven. Resume support uses `--resume`; only compatible saved scan results are reused.

To continue an interrupted calculation, use `.\run.ps1 --resume` on Windows or `sh run.sh --resume` on Linux/macOS. A second fresh invocation refuses to overwrite an existing numerical run; choose a new `--output` to start again. Different modes must use different output directories. Per-scan spectra and selection evidence are saved as well as figures, so allow free disk space for the full audit output; `results/` is excluded from Git.

Figures show the BO dynamic spectrum in gray and selected candidates in color. Colors distinguish visual tracks, not independently established physical-mode identities. Known auxiliary poles and zero/constraint roots are excluded from the plotted background. See [reproducibility](docs/reproducibility.md) for settings and the distinction between a fresh solve, a cached redraw and physical validation.

## Source layout

```text
reproduce.py          Common command-line entry point
bo_t/                 Case configuration, reproduction and plotting
boarbitrary_py/       Numerical solver and selector internals
inputs/               Numerical case inputs and distribution coefficients
data/frozen/          Compact frozen arrays and provenance for cached redraw
tests/                Release tests
docs/                 Reproduction protocol
LICENSES/             Unmodified upstream license notices
results/              Generated files (not tracked)
```

This is an editable source-tree installation. Keep `inputs/` and `data/` beside the code; a standalone wheel is not the supported delivery format.

## Tests

```sh
python -m pip install -r requirements.txt -e ".[test]"
python -m pytest tests -q
```

Continuous integration runs the tests, the seven cached figures and an ES1D smoke calculation. It does not rerun all full scans or the manuscript speedup tables.

All seven complete scans (1,061 points) have been rerun locally and checked
against the frozen manuscript predictions. Root counts agree at every point;
all dynamic and selected roots match within the predefined normalized tolerance
of 1e-4. See the [release verification record](docs/validation.md) for measured
errors, installation tests and the distinction from independent physical
validation.

The distribution also includes a standard-library checksum check:

```sh
python verify_release.py
```

It checks the packaged source and data against `FILE_MANIFEST.json`. Intentional edits will change these checksums. `.gitattributes` preserves the checksummed bytes across Git checkouts on different platforms.

## Provenance and licensing

BO-T builds on [PDRK](https://github.com/hsxie/pdrk), [BO](https://github.com/hsxie/bo) and [BO-Arbitrary](https://github.com/hsxie/boarbitrary). Their notices are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`.

The authors have not yet selected a license for new BO-T contributions. The included upstream licenses do not constitute a license declaration for the entire project. A project license must be settled before this release is advertised as open-source software.

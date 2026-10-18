# Release verification

Verification was performed on Windows with Python 3.14.5, NumPy 2.4.6,
SciPy 1.17.1, Matplotlib 3.10.9 and threadpoolctl 3.6.0. Numerical calculations
used one BLAS thread. This is a reproducibility check, not a speedup benchmark
or an independent validation of the underlying physical models.

## Installation and interface

- The packaged tree was extracted to a different directory containing spaces.
  A clean virtual environment installed it without the research workspace.
- All seven cached figures and all eight midpoint smoke calculations ran from
  that relocated tree. The signed-EM example has one midpoint for each sign.
- The PowerShell launcher created its own virtual environment, installed the
  pinned direct dependencies and completed an ES1D smoke calculation.
- Smoke results resumed successfully after integrity checks. A full numerical
  campaign was also deliberately interrupted and resumed; committed scans
  were verified, and uncommitted temporary scans were not reused.
- The release test suite has 53 passing tests, including frozen-data integrity,
  all seven visual-track audits, mode isolation, failure propagation and stale
  resume rejection.
- The seven cached PNG files are byte-identical to the current manuscript
  PNGs in the reference environment. Their PDF renderings were also inspected.
- Upstream license files are unchanged. No new BO-T license has been assigned.

Linux and Windows CI jobs are supplied, but hosted GitHub CI has not been run
for this local release. A configured workflow is not a claim of tested hosted
platforms. The four direct dependencies are pinned; this is not a complete
transitive environment lock.

## Full-scan numerical regression

All seven cases were recomputed from the bundled numerical inputs, including
source eigensolves and root selection at all 1,061 scan points. Afterward, the
common entry point verified every committed scan through `--resume` and rendered
all seven figures. No selector settings were adjusted during this verification.

| Case | Scan points | Dynamic roots | Selected candidates | Maximum normalized dynamic-root error | Maximum normalized selected-root error |
| --- | ---: | ---: | ---: | ---: | ---: |
| ES1D beam | 50 | 800 | 207 | 0 | 0 |
| Umeda ring-beam | 117 | 63,795 | 1,689 | 6.74e-10 | 1.48e-10 |
| Signed EM | 500 | 26,970 | 6,092 | 0 | 0 |
| HH ring-beam | 117 | 78,770 | 1,999 | 1.65e-11 | 2.40e-13 |
| Firehose | 46 | 22,476 | 3,077 | 0 | 0 |
| Muschietti beam | 131 | 42,938 | 1,633 | 1.73e-8 | 4.08e-13 |
| Loss-cone mirror | 100 | 23,000 | 1,845 | 0 | 0 |

Counts agree at every scan point, and all roots match one-to-one within the
predefined normalized tolerance of 1e-4. Zero denotes identical stored complex
frequencies under unordered matching in the reference environment. The
[machine-readable record](../data/validation.json) includes full-precision
errors, unmatched counts, environment information and run fingerprints.

The full campaign was split into case groups after testing interruption and
restart. Its accumulated scan times are operational records, not timings for
the manuscript speedup table. No speedup claim is inferred from this campaign.

## Interpretation

Fresh-versus-frozen comparison is performed only after predictions have been
committed. It matches complex roots one-to-one at each scan coordinate using
absolute frequency difference divided by one plus the reference root's
magnitude. Count mismatches and unmatched roots are reported separately.

The comparison is against the manuscript's frozen numerical predictions,
not paper-digitized curves. It confirms preservation of this calculation;
it does not convert stable candidates into independently certified physical
modes. The HH ring-beam policy remains a post-hoc development choice.

Gray-spectrum counts exclude declared auxiliary poles and zero/constraint
roots. Candidate counts are before plotting-window clipping. Rendered colors
only group adjacent points visually and do not add or interpolate roots.

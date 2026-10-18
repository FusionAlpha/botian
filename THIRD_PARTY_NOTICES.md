# Third-party notices

The numerical implementation and input capsules draw on the BO/PDRK program family. Upstream license texts are included without modification. They apply to the respective upstream material; no license for new BO-T contributions has yet been selected.

## BO-Arbitrary

Source: [hsxie/boarbitrary](https://github.com/hsxie/boarbitrary).

Notice: Copyright (c) 2025, Hua-sheng XIE. The distribution's BSD 3-Clause license is reproduced in [LICENSES/BO-Arbitrary.txt](LICENSES/BO-Arbitrary.txt).

Relevant material includes the Hermite-Hermite response and matrix formulation, J-pole coefficient tables, MATLAB-corresponding moment recurrences, case inputs and distribution-coefficient provenance used by the Python implementation. The original source also contains this historical notice in its main driver:

> Copyright@2018-2021 ENN Fusion Technology R&D Center (ENN-FTRC, Huasheng Xie, xiehuasheng@enn.cn, huashengxie@gmail.com)

## BO/PDRK case sources

Sources: [hsxie/bo](https://github.com/hsxie/bo) and [hsxie/pdrk](https://github.com/hsxie/pdrk).

The bundled legacy numerical inputs originate from the BO 190305/190307 source distribution. Its notice is reproduced in [LICENSES/BO-PDRK.txt](LICENSES/BO-PDRK.txt): Copyright &lt;2014-2019&gt; &lt;Hua-sheng XIE, huashengxie@gmail.com&gt;, with three-clause BSD redistribution conditions.

Input provenance and numerical overrides are retained in the case capsules. The public-facing case names do not rename or remove the attribution of the underlying formulations.

## Python dependencies

NumPy, SciPy, Matplotlib and threadpoolctl are installed separately through the package manager. Their distributions carry their own license notices. SciPy supplies the Faddeeva-function implementation used by the Python plasma-response evaluator; this tree does not vendor a separate implementation.

## Scope of this release

Publisher reference PDFs, extracted paper graphics and compiled MATLAB Runtime artifacts are not included. Numerical inputs and frozen calculation outputs are kept separate from external paper references. The presence of upstream BSD notices must not be interpreted as assigning a project-wide license to BO-T.

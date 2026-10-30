# Analysis code — "How Early Can Teachers Act? Warning-Week Selection for At-Risk Nursing Students in a Bioscience Course"

Code accompanying the manuscript submitted to *Medical Teacher*. This
repository contains the analysis code only. **No student-level data are
included.** De-identified analysis datasets are available from the
corresponding author upon reasonable request, subject to institutional data
governance approval; raw LMS event logs contain student identifiers and are
not shared.

## Contents

### `01_etl/` — Data processing (LMS ETL, v7.15.34)

The pipeline that converts raw iClass LMS exports and grade files into the
de-identified, analysis-ready records used in the study: loading and parsing
LMS exports, de-identification (`lms_etl/modules/02_anonymize.py`; student IDs
are replaced by salted SHA-256 codes, with the salt read at run time from an
external `config/salt.txt` that is **not** part of this repository), feature
engineering, grade merging and export. `01_etl/README_ETL.md` is the ETL's own
documentation; `requirements.txt` lists its Python dependencies. The `config/`
and `raw_data/` folders contain only README placeholders. (The ETL's own
README refers to its full development package, including a test suite and a
changelog, which are not part of this deposit.)

### `02_ews_pipeline/` — Model fitting and out-of-fold scoring

- `ews_lib.py` — shared library: the behaviour-only predictors (the
  rule-based BAS composite M1′ and XGBoost M2′), percentile ranking, and
  student-cluster bootstrap confidence intervals.
- `ews_experiment.py` — the experiment driver: assembles the
  leave-one-semester-out out-of-fold scores and the M3′ blend (the mean of
  the M1′ and M2′ percentile ranks). Its built-in `wstar()` implements the
  **first version** of the warning-week selection rule (Week 16 reference;
  a relative correlation criterion). The **final documented rule** — earliest
  week whose upper 95% confidence bounds stay within an AUC loss of 0.02 and
  a correlation loss of 0.03 of the Week 12 reference, with at least four
  tutoring weeks remaining — is implemented in
  `03_supplementary_analyses/phase2/analysis/h_lib.py` (constants `D_AUC`,
  `D_R`; functions `qualifies`, `earliest_week`) and is the version applied
  by the re-selection scripts in `03_supplementary_analyses/`. The version
  history is documented in the manuscript's Methods (Specification history).

### `03_supplementary_analyses/` — Supplementary analyses

Scripts for the analyses reported in the manuscript's Supplementary Material,
grouped by the batch in which they were run (`phase1/`, `phase2/`, `round3/`,
`round4/`): reference-week sensitivity, within-semester (deployment)
re-selection, baseline comparators and incremental value, lead-time and
tutoring-capacity calculations, era and class-type heterogeneity, the
blinded 1142 holdout checks, and the pre-examination-window (placebo-arm)
analyses. Shared helpers are in `phase2/analysis/` and `round3/pa_common.py`.

## Notes

- The scripts read de-identified input files (learning-behaviour features,
  grades and frozen out-of-fold scores) that are not distributed with this
  code, for the reasons stated above; file and folder names in the scripts
  reflect the original analysis workspace.
- Figure-drawing scripts for the manuscript's figures are not part of this
  deposit.

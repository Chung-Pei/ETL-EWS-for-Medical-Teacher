# Analysis code — "How Early Can Teachers Act? Post-Midterm Warning Lists for Nursing Students in a Bioscience Course"

Code accompanying the manuscript submitted to *Medical Teacher*. The study asks, under a prespecified and locked selection rule, which week a behaviour-based early-warning list should be compiled in a nursing microbiology and immunology course, and which score and list size it should use.

## What is public, and what is not

This repository contains the analysis code only: data processing, model fitting and the policy selection rule.

De-identified analysis datasets are available from the author upon reasonable request, subject to institutional data governance approval; raw LMS event logs contain student identifiers and are not shared. One de-identified record identifier that appears in the Round 5 to 7 audit code has been replaced by the placeholder WITHHELD in this public copy, to prevent cross-linkage with datasets released under approved requests. Student-level data, including the frozen out-of-fold scores, are not publicly deposited. Because the frozen scores are not deposited and the public pipeline reconstructs the frozen feature values only approximately, the reported results cannot be reproduced exactly from the code alone. "Auditable", as used in the manuscript, refers to the decision trail: the rule, its versions, the lock and the disclosed deviations are documented and can be inspected. It does not mean that the numerical results can be recomputed from the public materials. The analysis plan and the lock manifest are available from the author upon reasonable request.

## Repository layout

- `01_etl/` — ETL pipeline: converts raw LMS exports and grade books into the anonymized analysis datasets (feature engineering, grade merging, anonymization).
- `02_ews_pipeline/` — the frozen early-warning pipeline: the three grade-free predictors (rule-based score BAS, XGBoost, and their percentile blend), leave-one-semester-out validation, and the locked week-selection rule.
- `03_supplementary_analyses/phase1/` — first supplementary analyses: reference-week sensitivity (H-04), within-semester (deployment) definition (H-09), and a Table 1 audit (H-15).
- `03_supplementary_analyses/phase2/` — the H-series analyses: calendar audit (H-01), baselines (H-02), lead time and tutoring capacity (H-03), examination-window arms (H-07), the Semester 1142 holdout evaluation (H-09_1142), and a denominator check (H-15.3).
- `03_supplementary_analyses/round3/` — Round 3 re-runs: class-only and temporal-prior baselines (H-02), examination-window arm rebuilds (H-07), and the deployment-definition retraining and selection (H-09 part 4).
- `03_supplementary_analyses/round4/` — Round 4 examination-window analyses: arm analyses and the placebo-contrast arm (H-07).
- `03_supplementary_analyses/round5/` — Round 5 sensitivity analyses (Supplementary Tables 14–20; see the mapping below). `r5_lib.py` is the shared library for this and later rounds.
- `03_supplementary_analyses/round6/` — Round 6: re-computation of the baseline increments under the bonus-free (Variant 1) outcome (Supplementary Table 18, Panel B).
- `03_supplementary_analyses/round7/` — Round 7 supplementary experiments: training-set size pairing, BAS weight perturbation, era/track re-selection, refitting distribution, and interval widths.
- `03_supplementary_analyses/probe/` — the phase-week probe: Week 9 against Week 10 in the 16+2 semesters under the deployment definition (Supplementary Table 21).

Each round folder is self-contained as delivered, so shared libraries (notably `r5_lib.py`) appear in more than one folder.

## Which code produces which supplementary table

| Supplementary table | Code |
|---|---|
| Table 13. Selection under the deployment definition | H-09 deployment retraining and selection — `03_supplementary_analyses/round3/h09p4_*` |
| Table 14. Performance against the final-examination outcome | Round 5 — `round5/r5_01_final_exam.py` |
| Table 15. Selection frequencies under resampling | Round 5 — `round5/r5_03_selection_freq.py` (frozen scores) and `round5/r5_09_refit.py`, `round5/r5_09_analyze.py` (refitting-inclusive); Panel C distribution summaries from Round 7 — `round7/r7_04_refit_pooled.py` |
| Table 16. Forward-chaining validation | Round 5 — `round5/r5_05_forward.py` |
| Table 17. Sensitivity analysis admitting the quiz pass rate (B3) as a candidate | Round 5 — `round5/r5_04_b3_candidate.py` |
| Table 18. Sensitivity of the selection to the evaluation set and outcome construction | Round 5 — `round5/r5_08_outcome.py`, `round5/r5_12_drop1132.py`, `round5/r5_06_07_per_semester_retake.py`; Panel B increments under the bonus-free outcome recomputed in Round 6 — `round6/r6_02_variant1.py` |
| Table 19. Reactivity probe | Round 5 — `round5/r5_11_reactivity.py` |
| Table 20. Per-semester discrimination and grade-book adjustments | Round 5 — `round5/r5_06_07_per_semester_retake.py`, `round5/r5_10_gradebook.py` |
| Table 21. Phase-week probe (Week 9 vs Week 10, 16+2 semesters) | Probe — `probe/pw_*.py` |
| Table 22. BAS weight perturbation | Round 7 — `round7/r7_02_bas_perturb.py` |

Round 7 also produced Supplementary Table 1, Panel B (single-semester training pairing — `round7/r7_01_trainsize.py`); its era/track re-selection (`round7/r7_03_era_track.py`) and interval-width analysis (`round7/r7_05_width.py`) are reported in the main text and in the text accompanying Supplementary Table 15, with no separate supplementary table. The Round 5 audit of the Week-10 deployment list appears in Supplementary Table 6, Panel B and Supplementary Table 10, Panel D3 (`round5/r5_02_week10_lists.py`).

## Environment

Analyses were run in Python (3.12/3.13) with pandas 3.0.2/3.0.6, scikit-learn 1.8.0/1.9.1 and XGBoost 3.4.1 (CPU build), as recorded in the project ETL changelog; the Supplementary Material's Computational Notes specify XGBoost 3.4.1 for the refitting runs. Exact package versions varied slightly between analysis rounds; the frozen pipeline's documented reproduction tolerance is ±0.003 AUC.

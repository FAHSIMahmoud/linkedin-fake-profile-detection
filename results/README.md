# Results

Files produced by the notebooks for the paper. All metrics are computed on held-out test splits;
precision, recall and F1 refer to the positive class unless a column name says otherwise.

| file | produced by | contents |
|---|---|---|
| `results_all.csv` | 06, merging 00-05 | one row per run: task, seed, search strategy, variant, test metrics, confusion counts, selected parameters, CV objective, number of models evaluated, runtime |
| `results_summary.csv` | 06 | mean and SD over three splits per task, strategy and variant |
| `results_embeddings.csv` | 07 | embedding screen: one row per variant, task and seed |
| `embedding_summary.csv` | 06 | embedding screen: mean per task and variant |
| `summary.txt` | 06 | completeness check, reproducibility check, HGS against random and exhaustive search, ablations, paired embedding comparisons |
| `results_bestemb.csv` | 09 | tag-direction projection (V5) under HGS, one row per task and seed |
| `bestemb_summary.csv`, `bestemb_summary.txt` | 09 | V5 against STE under HGS, paired by split, with the published figures |
| `results_peerj.csv` | 08 | four-class task: one row per embedding, split protocol and seed, with per-class metrics and confusion matrices |
| `peerj_summary.txt` | 08 | four-class results against Mohiuddin and Almogren (2025) |

| paper item | source |
|---|---|
| Table II (experimental effort) | all result files: run counts, `n_models`, `seconds` |
| Table III | `results_all.csv`, `strategy == "HGS"`, `variant == "main"` |
| Table IV, Fig. 5 | Table III values, `results_bestemb.csv` (projection), `results_peerj.csv` (T4) |
| Table V, Fig. 6 | `results_all.csv`, HGS, Random and Grid rows |
| Table VI | `results_all.csv`, seed 42 (`cv_fitness`, `n_models`, `f1_pos`) |
| Table VII | `results_all.csv`, HGS rows, `params_fitted` |
| Table VIII, Fig. 7 | `results_embeddings.csv` |
| Table IX | `results_all.csv`, `variant` in `no_tag_correction`, `no_full_name` |
| Table X, Fig. 8 | `results_bestemb.csv` against the HGS rows of `results_all.csv` |
| Table XI, Fig. 9 | `results_peerj.csv` (`confusion` column for Fig. 9) |
| Fig. 3 | computed from the OWA definition (no result file) |
| Fig. 4 | `results_all.csv`, T3 HGS rows, `TP`, `FP`, `FN`, `TN` summed over seeds |

On the four-class task, the unstratified split drawn with seed 42 gives test counts of 349, 147, 117
and 107 profiles, the counts reported by Mohiuddin and Almogren; `results_peerj.csv` records them in
the `n_test_*` columns.

Embedding variants: V0 STE (tag subtraction), V1 plain mean, V2 R1 renormalization, V3 R2 with a
global mean, V4 R2 with per-section means, V5 projection off the tag direction, V6 ZCA whitening,
V7 STE plus cross-section cosine coherence, V8 V4 plus coherence, V9 STE plus L1 distances between
sections.

## Files added in version 1.2.0

| file | produced by | contents |
|---|---|---|
| `results_confirm.csv`, `confirm_summary.txt` | 10 | pre-registered confirmation on seeds 1001-1010: one row per variant and seed; decision under Holm correction |
| `stats_report.txt`, `stats_claims.csv`, `stats_intervals.csv` | 11 | rebuild check (52 of 53 stored runs reproduced exactly), bootstrap intervals, paired tests, tests against published figures |
| `results_baselines.csv`, `baselines_report.txt` | 12 | re-implemented published detectors on our test profiles, paired tests |
| `results_objective.csv`, `objective_report.txt` | 13 | objective ablation |
| `results_emscad.csv`, `emscad_report.txt` | 14 | EMSCAD external validation |
| `results_groupsplit.csv`, `groupsplit_report.txt` | 15 | template-group splits |

Revised paper (version 2) items: Table III and IV and Fig. 4 from `stats_intervals.csv` and `stats_claims.csv`;
Table V from `results_baselines.csv`; Table VI from `groupsplit_report.txt`; Fig. 6 from `stats_report.txt`,
`confirm_summary.txt`, `groupsplit_report.txt` and `emscad_report.txt`; Table VII from `stats_report.txt`,
`emscad_report.txt` and `objective_report.txt`; Table IX from `emscad_report.txt`.

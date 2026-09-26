# Results

Files produced by the notebooks for the paper. All metrics are computed on held-out test splits;
precision, recall and F1 refer to the positive class unless a column name says otherwise.

| file | produced by | used in the paper |
|---|---|---|
| `results_all.csv` | 06, merging 00-05 | one row per run: task, seed, search strategy, variant, test metrics, confusion counts, selected parameters, CV objective, number of models evaluated, runtime |
| `results_summary.csv` | 06 | Table II (HGS rows), mean and SD over three splits |
| `results_embeddings.csv` | 07 | one row per embedding variant, task and seed |
| `embedding_summary.csv` | 06 | Table VI |
| `summary.txt` | 06 | completeness check, reproducibility check, HGS against random and exhaustive search, ablations, paired embedding comparisons |

| paper item | source |
|---|---|
| Table II | `results_all.csv`, `strategy == "HGS"`, `variant == "main"` |
| Table IV, Fig. 5 | `results_all.csv`, HGS and Random rows |
| Table V | `results_all.csv`, seed 42, HGS / Grid / Random rows (`cv_fitness`, `n_models`, `f1_pos`) |
| Fig. 4 | `results_all.csv`, T3 HGS rows, `TP`, `FP`, `FN`, `TN` summed over seeds |
| Table VI, Fig. 6 | `results_embeddings.csv` |
| ablations | `results_all.csv`, `variant` in `no_tag_correction`, `no_full_name` |

Notebooks 08 and 09 write their results next to these files when run (`results_peerj.csv`,
`results_bestemb.csv` and their summaries).

Embedding variants: V0 STE (tag subtraction), V1 plain mean, V2 R1 renormalization, V3 R2 with a
global mean, V4 R2 with per-section means, V5 projection off the tag direction, V6 ZCA whitening,
V7 STE plus cross-section cosine coherence, V8 V4 plus coherence, V9 STE plus L1 distances between
sections.

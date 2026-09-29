[![DOI](https://zenodo.org/badge/1389185997.svg)](https://doi.org/10.5281/zenodo.22978171)

# Fake LinkedIn Profile Detection across Manual and LLM-Generated Threats

Code and results for the paper *Fake LinkedIn Profile Detection across Manual and LLM-Generated
Threats: Tag-Direction Projection and Fuzzy OWA Hyperparameter Search* (M. Fahsi, N. Mahammed,
A. Kourtiche, L. Septem Riza, C. Mouilah), submitted to *Advances in Electrical and Computer
Engineering*.

The repository contains every notebook used in the paper, the result files behind each table and
figure, and a script that regenerates the figures from those results.

## What the study does

One detection pipeline is evaluated on four tasks defined from the labels of the public LinkedIn
corpus of Ayoobi et al. (2023):

| task | negative class | positive class | protocol |
|---|---|---|---|
| T1 | legitimate (code 0) | manual fake (1) | Gulati et al. 2026: 1,260/420 train, 540/180 test |
| T1b | legitimate, 600 drawn | manual fake (1) | Ayoobi et al. 2023: 420/420 train, 180/180 test |
| T2 | written by people (0, 1) | ChatGPT-generated (10, 11) | no published counterpart |
| T3 | legitimate (0) | any fake (1, 10, 11) | Gulati et al. 2026, GPT-3.5 retraining scenario |

Codes 10 and 11 are both ChatGPT-generated profiles, built from the statistics of legitimate and of
fake profiles respectively, as the dataset documentation states.

The pipeline encodes three profile sections with frozen `roberta-base`, adds 15 section counts, and
trains a logistic regression whose hyperparameters are chosen by Hunger Games Search (HGS) against a
fuzzy ordered weighted average (OWA) of cross-validated F1, recall and ROC area, with a penalty on the
gap between training and cross-validated scores. Every task is run on three stratified splits
(seeds 42, 7, 123) and compared with random search at the same number of evaluations and, for seed 42,
with exhaustive search over the 336 distinct models of the grid.

## Data

The corpus is **not included**. Its official repository
([navid-aub/LinkedIn-Dataset](https://github.com/navid-aub/LinkedIn-Dataset)) has no licence file and
asks users to cite Ayoobi et al. (2023). Download and verify it with

```bash
python scripts/download_data.py
```

which saves `data/LinkedIn_Dataset.pcl` and checks its SHA-256 and label counts. The notebooks repeat
the checksum test and report whether the file matches the release used here. See
[`data/README.md`](data/README.md).

## Running the notebooks

The notebooks run on Google Colab (data and results on Google Drive) or locally from the
`notebooks/` folder (data in `data/`, results in `results/run/`). Both locations can be overridden
with the environment variables `LINKEDIN_DATA_DIR` and `LINKEDIN_OUT_DIR`.

Run **notebook 00 first, on a GPU**: it computes the RoBERTa section embeddings once and caches them.
All other notebooks load the cache and run on a CPU. Every model evaluation is saved as it completes,
so an interrupted notebook resumes where it stopped when re-run from the top.

| notebook | runs | runtime | writes |
|---|---|---|---|
| `00_main_study_T1_T1b` | embeddings; replay of the original T2 run; T1 and T1b with HGS, random and exhaustive search | GPU; about 13 h, most of it the two exhaustive searches | `results_raw.csv` |
| `01_T2_seed42` | T2, seed 42, HGS and random search | about 1.3 h | `results_T2_seed42.csv` |
| `02_T2_seeds7_123` | T2, seeds 7 and 123 | about 1.5 h | `results_T2_seeds7_123.csv` |
| `03_T2_exhaustive_grid` | T2, seed 42, all 336 models | about 13.5 h | `results_T2_grid.csv` |
| `04_T3_gulati_protocol` | T3, three seeds, split stratified on the raw code | about 1.6 h | `results_T3.csv` |
| `05_ablations` | tag subtraction removed; name removed from the introduction | about 1 h | `results_ablations.csv` |
| `06_summary` | merges 00-05 and 07; completeness and reproducibility checks | seconds | `results_all.csv`, `results_summary.csv`, `embedding_summary.csv`, `summary.txt` |
| `07_embedding_screen` | ten embedding formulations x four tasks x three seeds, fixed classifier | under 5 min | `results_embeddings.csv` |
| `08_peerj_4class` | four-class task of Mohiuddin and Almogren (2025) | about 4 h | `results_peerj.csv`, `peerj_summary.txt` |
| `09_best_embedding_hgs` | best screen variant under the full HGS pipeline, paired with STE | about 2.6 h | `results_bestemb.csv`, `bestemb_summary.txt` |

### Notebooks added in version 1.2.0 (statistical checks and external validation)

| notebook | purpose | runtime | writes |
|---|---|---|---|
| `10_concern1_new_seed_confirmation` | pre-registered test of the projection (V5) against STE on T3, ten new seeds (1001-1010), F1 and ROC area, Holm correction; protocol in `preregistration_confirm.json` (commit 0f82b13) | about 5 h | `results_confirm.csv`, `confirm_summary.txt` |
| `11_concern2_statistics` | rebuilds the stored models, bootstrap intervals, paired tests (DeLong, McNemar, corrected resampled t), tests against the published figures | 10-20 min | `stats_report.txt`, `stats_claims.csv`, `stats_intervals.csv` |
| `12_concern3_published_baselines` | re-implemented detectors of Gulati et al. (XGBoost, CatBoost) and Ayoobi et al. (five classifiers) on our test profiles | about 16 h, GPU for CatBoost | `results_baselines.csv`, `baselines_report.txt` |
| `13_concern4_objective_ablation` | the OWA objective against three alternatives (no gap penalty, arithmetic means, mean CV F1) | several hours | `results_objective.csv`, `objective_report.txt` |
| `14_concern5_job_postings_validation` | external validation on EMSCAD (fraudulent job postings): composition test, projection, duplicate leakage, HGS against random search | about 21 h, GPU once | `results_emscad.csv`, `emscad_report.txt` |
| `15_template_group_split` | manual fakes that reuse text templates: memorization diagnostic and group-aware re-run of T1, T1b and T3 | about 5 h | `results_groupsplit.csv`, `groupsplit_report.txt` |

Notebook 14 needs `fake_job_postings.csv` (EMSCAD, Vidros et al. 2017, available on Kaggle as
"Real or Fake Job Posting Prediction") in the data folder; its results go to a separate folder
(`EMSCAD_OUT_DIR`, default `emscad_validation/`).

Runtimes are the sums of the times recorded in the result files. They were taken on shared Colab machines and indicate orders of magnitude.

## Results

[`results/`](results/) holds the files produced by notebooks 00-15 for the paper, and
[`results/README.md`](results/README.md) maps each file to the tables and figures it supports. To
redraw the figures:

```bash
python scripts/make_figures.py        # figures of notebooks 00-09
python scripts/make_figures_v2.py     # Fig. 4 and Fig. 6 of the revised paper
```

## Reproducibility

- Every split, subsample and search is seeded. Rerunning the T2 seed-42 search in notebook 01
  reproduced the F1 and accuracy of the original run to four decimals; `summary.txt` records the check.
- Standardization statistics and any statistic of an embedding variant are fitted on the training
  split only. The test split is used once, after the search.
- Notebook 00 opens with a *faithful replay* of the original single-split T2 run, which used scaling
  fitted on all rows. The replay exists to check the environment; every other experiment uses
  training-split scaling.
- Notebook 08 adapts the fitness to four classes: the original objective computes the ROC area only
  for binary tasks, so on four classes it would silently drop that criterion. The adapted version uses
  macro F1, macro recall and macro one-vs-rest ROC area.

## Requirements

Python 3.10 or later and the packages in [`requirements.txt`](requirements.txt). The experiments
were run on Google Colab in 2026.

## Citation

If you use this code, please cite it through its Zenodo DOI, 10.5281/zenodo.22978171 (all versions), the
paper (details will be added on publication) and the dataset:

> N. Ayoobi, S. Shahriar, and A. Mukherjee, "The looming threat of fake and LLM-generated LinkedIn
> profiles: challenges and opportunities for detection and prevention," in *Proc. 34th ACM Conf.
> Hypertext and Social Media (HT '23)*, 2023. doi:10.1145/3603163.3609064

A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

## Licence

The code is released under the MIT licence ([`LICENSE`](LICENSE)). The licence does not cover the
dataset, which is distributed by its authors.

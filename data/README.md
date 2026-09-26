# Data

This folder is empty on purpose. The LinkedIn corpus belongs to Ayoobi et al. (2023) and is
distributed from [github.com/navid-aub/LinkedIn-Dataset](https://github.com/navid-aub/LinkedIn-Dataset).
That repository has no licence file, so the file is not copied here.

```bash
python scripts/download_data.py
```

downloads `LinkedIn_Dataset.pcl` into this folder and verifies it.

| property | value |
|---|---|
| file | `LinkedIn_Dataset.pcl` (pandas pickle, 20,786,720 bytes) |
| SHA-256 | `6d67c8015fceee06226491a8b0683bfc01e220416fed99e7f2d91aca27ba2e1a` |
| shape | 3,600 rows x 39 columns |
| label column | `Label` |

| code | meaning (dataset documentation) | profiles |
|---|---|---|
| 0 | legitimate LinkedIn profile | 1,800 |
| 1 | fake profile created manually | 600 |
| 10 | ChatGPT-generated profile, built from legitimate profiles' statistics | 600 |
| 11 | ChatGPT-generated profile, built from fake profiles' statistics | 600 |

The file is a Python pickle. Unpickling can execute code, so load it only after its checksum has been
verified, as the download script and the notebooks do.

If you cite results obtained with this corpus, cite Ayoobi et al. (2023), doi:10.1145/3603163.3609064.

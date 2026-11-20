# Data-Driven Site Characterization

Machine learning models for site characterization from cone penetration test (CPT) data.

Builds on the merged CPT database (Dataset 1: global CPT streams, Dataset 2: coupled CPT + laboratory records) described in the companion *Data in Brief* article ([doi:10.1016/j.dib.2026.113428](https://doi.org/10.1016/j.dib.2026.113428)). The data are archived on Zenodo ([doi:10.5281/zenodo.21793782](https://doi.org/10.5281/zenodo.21793782)). This repo will hold the data-loading/preprocessing pipeline and the ML models trained on that database.

---

## Structure

```
├── data/               - CPT database (not tracked directly; large files via git-lfs)
├── docs/                - notes, writeups
├── graphics/             - output figures
├── models/               - trained/saved model artifacts
├── src/                  - data loading, preprocessing, model code
├── requirements.txt
└── LICENSE
```

## Data

The database is archived on Zenodo (version V1, published 4 August 2026, license CC BY 4.0):
<https://doi.org/10.5281/zenodo.21793782>

| File | Content | Size |
|---|---|---|
| `Dataset1_CPT_database.parquet` | Dataset 1: CPT, CPTu, SCPT and SCPTu soundings (11,684 soundings, one row per depth increment) | 311.2 MB |
| `Dataset2_Coupled_Geotechnical_CPT.parquet` | Dataset 2: 1,544 records that couple CPT data with laboratory-derived soil parameters | 101.7 kB |

Both files are Parquet files. The article describes the columns, the screening and the limitations of each dataset.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows — or: source venv/bin/activate
pip install -r requirements.txt
```

Download the two Parquet files from Zenodo (see [Data](#data)) and place them in `data/` before running anything in `src/`.

## Dependencies

`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `openpyxl`, `jupyterlab` — see `requirements.txt` (will grow as the modeling work here does).

## Citation

If you use the data or this repository, please cite the article and the dataset.

**Article** (accepted 9 November 2026, in press; volume and pages to be added):

> Felić, H., Yaz, A. and Tschuchnigg, F. (2026) 'A Dataset for Data-Driven Soil Characterization: Cone Penetration Test Soundings and CPT-Coupled Laboratory Parameters (V1)', *Data in Brief*. Available at: https://doi.org/10.1016/j.dib.2026.113428.

**Dataset** (Zenodo, V1):

> Felić, H., Yaz, A. and Tschuchnigg, F. (2026) *A Dataset for Data-Driven Soil Characterization: Cone Penetration Test Soundings and CPT-Coupled Laboratory Parameters (V1)*. Zenodo. Available at: https://doi.org/10.5281/zenodo.21793782.

BibTeX:

```bibtex
@article{Felic2026DataInBrief,
  author  = {Feli{\'c}, Haris and Yaz, Ali and Tschuchnigg, Franz},
  title   = {A Dataset for Data-Driven Soil Characterization: Cone Penetration Test Soundings and CPT-Coupled Laboratory Parameters (V1)},
  journal = {Data in Brief},
  year    = {2026},
  doi     = {10.1016/j.dib.2026.113428},
  note    = {In press}
}

@dataset{Felic2026Zenodo,
  author    = {Feli{\'c}, Haris and Yaz, Ali and Tschuchnigg, Franz},
  title     = {A Dataset for Data-Driven Soil Characterization: Cone Penetration Test Soundings and CPT-Coupled Laboratory Parameters (V1)},
  year      = {2026},
  version   = {V1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21793782},
  url       = {https://doi.org/10.5281/zenodo.21793782}
}
```

## License

The data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). For the license of the code, see `LICENSE`.

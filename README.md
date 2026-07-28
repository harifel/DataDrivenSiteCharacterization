# Data-Driven Site Characterization

Machine learning models for site characterization from cone penetration test (CPT) data.

Builds on the merged CPT database (Dataset 1: global CPT streams, Dataset 2: coupled CPT + laboratory records, Dataset 3: Geoprofile, Switzerland) described in the companion *Data in Brief* article. This repo will hold the data-loading/preprocessing pipeline and the ML models trained on that database.

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

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows — or: source venv/bin/activate
pip install -r requirements.txt
```

Place the CPT database (Parquet/xlsx) in `data/` before running anything in `src/`.

## Dependencies

`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `openpyxl`, `jupyterlab` — see `requirements.txt` (will grow as the modeling work here does).

## Citation

> [to be added upon publication of the companion Data in Brief article]

## License

See `LICENSE`.

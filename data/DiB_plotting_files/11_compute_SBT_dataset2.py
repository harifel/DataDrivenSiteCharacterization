"""
11_compute_SBT_dataset2.py

Recomputes the Robertson (2010) SBT zone for every row of Dataset 2
(sheet TRAINING_DP006_dataset_soilpara) from qc/fs/u2 using
sbt_classification.compute_sbt_zone(), and overwrites the existing SBT
column in-place with the new values.

Why overwrite instead of keep the old column: a dry-run comparison found
298/1442 existing SBT values (20.7%) disagree with the recomputation.
Several alternate formulas/implementations were tried (existing Rf(%)
column vs fs/qt-derived Rf, shapely-based vs custom point-in-polygon,
a_net=1.0 vs 0.8) and all matched the existing column WORSE than the
current sbt_classification.py default (fs/qt*100, a_net=0.8, custom
point-in-polygon) — so that default is used here. The mismatches
concentrate in the 510 blank-CPT_name rows and a few named soundings
(SalzburgCPTu29, OYSC09, HALC06), consistent with those soundings' qc/fs/u2
having been revised after SBT was last computed for them, rather than a
flaw in the classification approach. Decision (2026-07-28): recompute and
overwrite everywhere, treating current qc/fs/u2 as authoritative.

Edits the workbook in place with openpyxl (not pandas.to_excel) so
formatting, the other sheets (Sheet2, Sheet1), and formulas are preserved.
Saves to a new dated copy — the source file is left untouched.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

from sbt_classification import compute_sbt_zone

# ---- user-updatable paths / assumptions -----------------------------------
DB_DIR = Path(__file__).parent / ".." / "11_databases"
INPUT_XLSX = DB_DIR / "Dataset2_Coupled_Geotechnical_CPT_20260728.xlsx"
OUTPUT_XLSX = DB_DIR / "Dataset2_Coupled_Geotechnical_CPT_20260728_withSBT.xlsx"
SHEET_NAME = "TRAINING_DP006_dataset_soilpara"
CHART_DIR = Path(__file__).parent / "sbt_chart"

A_NET = 0.8  # net area ratio for the qt correction
# ----------------------------------------------------------------------------

df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

qc_mpa = pd.to_numeric(df["qc (MPa)"], errors="coerce").values
fs_mpa = pd.to_numeric(df["fs (MPa)"], errors="coerce").values
u2_mpa = pd.to_numeric(df["u2 (kPa)"], errors="coerce").fillna(0.0).values / 1000.0

if np.isnan(qc_mpa).any() or np.isnan(fs_mpa).any():
    raise ValueError("qc/fs must be present for every row — found NaNs, aborting.")

result = compute_sbt_zone(qc_mpa=qc_mpa, fs_mpa=fs_mpa, u2_mpa=u2_mpa, chart_dir=CHART_DIR, a_net=A_NET)
computed_zone = result["sbt_zone_computed"].to_numpy()

existing_sbt = pd.to_numeric(df["SBT"], errors="coerce")
both = existing_sbt.notna()
n_changed = int((both & (existing_sbt.astype("Int64") != pd.Series(computed_zone, dtype="Int64"))).sum())
n_filled = int((~both).sum())
print(f"Rows: {len(df)} | existing SBT changed: {n_changed}/{int(both.sum())} | previously blank, now filled: {n_filled}")

# --- edit the workbook in place (preserves formatting / other sheets) ------
wb = openpyxl.load_workbook(INPUT_XLSX)
ws = wb[SHEET_NAME]
header = [c.value for c in ws[1]]
sbt_col = header.index("SBT") + 1

n_rows = ws.max_row - 1
if n_rows != len(df):
    raise ValueError(f"Row count mismatch: sheet has {n_rows} data rows, dataframe has {len(df)}.")

for i, zone in enumerate(computed_zone):
    ws.cell(row=2 + i, column=sbt_col).value = int(zone)

DB_DIR.mkdir(exist_ok=True)
wb.save(OUTPUT_XLSX)
print(f"Saved {OUTPUT_XLSX}")

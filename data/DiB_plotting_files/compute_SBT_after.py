"""
dataset2_compute_sbt.py

Applies sbt_classification.compute_sbt_zone() to every row of Dataset 2
that has qc/fs (u2 optional, defaults to 0 where missing). This is
assumption-free: the SBT zone lookup needs only qc, fs, and u2 — no depth,
no groundwater depth, no assumed unit weight. Adds the computed columns
alongside the existing data, and — since Dataset 2 already has an "SBT"
column for 1325 records — cross-tabulates computed vs. existing zone as a
sanity check.

Unit conversion: Dataset 2 stores qc in MPa already, but fs and u2 in kPa.
sbt_classification expects qc/fs/u2 all in MPa (consistent units so
qt = qc + u2*(1-a_net) is valid), so fs and u2 are divided by 1000 here.

REQUIRES a `sbt_chart/` folder (Curve1.csv ... Curve30.csv) next to
sbt_classification.py for real zone numbers — without it every row
classifies as zone 0 ("Unclassified"). This run will show that placeholder
behavior until the chart folder is supplied.

Set COMPUTE_STRESS_EXTRAS = False below if you also want Qtn/Ic/alpha_m/M
(the constrained-modulus estimate) — those DO need depth + assumed gwt/
gamma, unlike the zone classification itself. Off by default since they're
a separate, optional calculation with its own assumptions baked in.

Update INPUT_XLSX below if the file moves.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sbt_classification import compute_sbt_zone, compute_stress_dependent_extras

# ---- user-updatable paths / assumptions -----------------------------------
INPUT_XLSX = "..\\11_databases\Dataset1_CPT_database_FINAL.csv"
SHEET_NAME = "TRAINING_DP006_dataset_soilpara"
CHART_DIR = Path(__file__).parent / "sbt_chart"  # Curve1.csv ... Curve30.csv
OUT_CSV = "Dataset1_CPT_database_FINAL_withSBT.csv"

A_NET = 0.8  # net area ratio for the qt correction — the only assumption
             # the zone classification itself actually needs

COMPUTE_STRESS_EXTRAS = False  # set True for Qtn/Ic/alpha_m/M (needs gwt/gamma)
GWT_M = 2.0
GAMMA_KNM3 = 18.0
# ----------------------------------------------------------------------------

df = pd.read_csv(INPUT_XLSX)
df = df.replace(r"^\s*$", np.nan, regex=True)

# rows that can be classified: need qc + fs; u2 defaults to 0 if missing
usable = df[["qc (MPa)", "fs (MPa)"]].apply(pd.to_numeric, errors="coerce")
mask = usable.notna().all(axis=1)
print(f"Rows with qc+fs available: {mask.sum()} / {len(df)}")

qc_mpa = pd.to_numeric(df.loc[mask, "qc (MPa)"], errors="coerce").values
# fs_mpa = pd.to_numeric(df.loc[mask, "fs (MPa)"], errors="coerce").values / 1000.0
fs_mpa = pd.to_numeric(df.loc[mask, "fs (MPa)"], errors="coerce").values 
u2_kpa = pd.to_numeric(df.loc[mask, "u2 (MPa)"], errors="coerce")
u2_mpa = u2_kpa.fillna(0.0).values
print(f"  of those, u2 was missing (defaulted to 0) for: {u2_kpa.isna().sum()} rows")

result = compute_sbt_zone(qc_mpa=qc_mpa, fs_mpa=fs_mpa, u2_mpa=u2_mpa, chart_dir=CHART_DIR, a_net=A_NET)
result.index = df.index[mask]
out = df.join(result)

if COMPUTE_STRESS_EXTRAS:
    depth = pd.to_numeric(df.loc[mask, "Depth (m)"], errors="coerce").values
    extras = compute_stress_dependent_extras(
        depth_m=depth, qt_mpa=result["qt_mpa"].values, fs_mpa=fs_mpa,
        gwt=GWT_M, gamma=GAMMA_KNM3,
    )
    extras.index = result.index
    out = out.join(extras)

if not CHART_DIR.is_dir():
    print(f"\nWARNING: chart folder not found at {CHART_DIR}")
    print("Every row will show sbt_zone_computed = 0 (Unclassified) until")
    print("Curve1.csv ... Curve30.csv are placed there.")

# --- cross-check against the existing SBT column where both are available -
both = out.dropna(subset=["SBT", "sbt_zone_computed"])
if len(both):
    agree = (both["SBT"].astype(int) == both["sbt_zone_computed"].astype(int)).mean()
    print(f"\nAgreement with existing SBT column: {agree:.1%} of {len(both)} overlapping rows")
    print("Cross-tab (rows = existing SBT, cols = newly computed):")
    print(pd.crosstab(both["SBT"].astype(int), both["sbt_zone_computed"].astype(int)))

out.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV} ({len(out)} rows, {out.shape[1]} columns)")
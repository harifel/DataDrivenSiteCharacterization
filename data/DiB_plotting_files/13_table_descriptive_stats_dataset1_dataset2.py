"""
13_table_descriptive_stats_dataset1_dataset2.py



Update the CONFIG block below (paths + column names) to match the
actual schema of two parquet files.

"""
#%%
from pathlib import Path
import sys
import pandas as pd
# --------------------------------------------------------------------------
# CONFIG -- adjust paths and column names to match your files/schema
# --------------------------------------------------------------------------

DB_DIR = Path(__file__).parent / ".." / "11_databases"

DATASET1_PARQUET = DB_DIR / "Dataset1_CPT_database.parquet"
DATASET2_PARQUET = DB_DIR / "Dataset2_Coupled_Geotechnical_CPT.parquet"


def print_dataset1_descriptive_stats(path):
    print(f"\nDataset 1: {path.name}")
    df = pd.read_parquet(path)

    features = ["Depth (m)", "qc (MPa)", "fs (MPa)", "Rf (%)", "u2 (MPa)", "Vs (m/s)"]

    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    raw_stats = df.describe().T.round(3)

    df = df[(df["qc (MPa)"] > 0) & (df["fs (MPa)"] > 0)]
    df = df[(df["Rf (%)"] >= 0.001) & (df["Rf (%)"] <= 31)]
    df = df[(df["qc (MPa)"] < 200)]
    df = df[df["u2 (MPa)"].isna() | ((df["u2 (MPa)"] > 0) & (df["u2 (MPa)"] < 8))]
    df = df[df["Vs (m/s)"].isna() | ((df["Vs (m/s)"] > 0) & (df["Vs (m/s)"] < 1500))]
    df = df[df["Depth (m)"].isna() | (df["Depth (m)"] >= 0)]

    clean_stats = df.describe().T.round(3)

    print("  Pre-processing statistics:")
    display(raw_stats)
    print("  Post-processing statistics:")
    display(clean_stats)
    return raw_stats, clean_stats


def print_dataset2_descriptive_stats(path):
    print(f"\nDataset 2: {path.name}")
    df = pd.read_parquet(path)

    su_cols = [
        "su_CU_undisturbed (kPa)",
        "su_CUe_undisturbed (kPa)",
        "su_DSS (kPa)",
        "su_UU_remoulded (kPa)",
        "su_UU_undisturbed (kPa)",
        "su_FCT (kPa)",
    ]

    raw_stats = df.describe().T.round(3)

    df = df.dropna(subset=["Depth (m)", "qc (MPa)", "fs (MPa)"], how="all")
    df = df[df["Depth (m)"].isna() | (df["Depth (m)"] >= 0)]
    df = df[df["qc (MPa)"].isna() | (df["qc (MPa)"] > 0)]
    df = df[df["fs (MPa)"].isna() | (df["fs (MPa)"] > 0)]
    df = df[df["Rf (%)"].isna() | df["Rf (%)"].between(0.001, 31)]
    # df = df[df["water_content (%)"].isna() | (df["water_content (%)"] < 100)]
    # df = df[df["gamma (kN/m3)"].isna() | (df["gamma (kN/m3)"] > 10)]
    # df = df[df["OCR (-)"].isna() | (df["OCR (-)"] < 10)]
    # df = df[df["e0 (-)"].isna() | (df["e0 (-)"] < 2)]
    for col in su_cols:
        df = df[df[col].isna() | (df[col] < 300)]

    clean_stats = df.describe().T.round(3)

    print("  Pre-processing statistics:")
    display(raw_stats)
    print("  Post-processing statistics:")
    display(clean_stats)
    return raw_stats, clean_stats

#%%
df1_raw, df1_clean = print_dataset1_descriptive_stats(DATASET1_PARQUET)
df2_raw, df2_clean = print_dataset2_descriptive_stats(DATASET2_PARQUET)
# %%

df1_raw.to_csv("Dataset1_descriptive_stats_raw.csv")
df2_raw.to_csv("Dataset2_descriptive_stats_raw.csv")

df1_clean.to_csv("Dataset1_descriptive_stats_clean.csv")
df2_clean.to_csv("Dataset2_descriptive_stats_clean.csv")

# %%
 
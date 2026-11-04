import re
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Restructure Dataset1_CPT_database_FINAL_withSBT.csv:
#   1. Add 'database' column (mapped source label) right after 'ID'
#   2. Drop 'qt_mpa' and 'rf_percent' (derived/intermediate columns not
#      meant for the public release)
#   3. Merge 'SBT' and 'sbt_zone_computed' into a single 'SBT' column,
#      preferring sbt_zone_computed (newer) where it has a value, falling
#      back to the original SBT where sbt_zone_computed is missing
#   4. Drop 'sbt_name_computed' (text label, redundant with the numeric zone)
# ---------------------------------------------------------------------------

INPUT = r"..\11_databases\Dataset1_CPT_database_FINAL_withSBT.csv"
OUTPUT = r"..\11_databases\Dataset1_CPT_database_FINAL_v2.csv"
SOURCE_COL = "Dataset"

TEST_MODE = False

df = pd.read_csv(INPUT, low_memory=False)
print("Original columns:", list(df.columns))
print("Original shape:", df.shape)

# ---------------------------------------------------------------------------
# 1. Recompute 'database' (same verified mapping used for Table 1 / Figure 1)
# ---------------------------------------------------------------------------
row_to_database = {
    1: "Nepal", 2: "Netherlands (Krogt)", 3: "Snakes", 4: "Baytown, Texas USA",
    5: "Hollywood, California USA", 6: "Emilia Romagna, Italy",
    7: "Terneuzen, Netherlands", 8: "Netherlands (1)", 9: "Premstaller",
    10: "Taiwan", 11: "Rhesi", 12: "NZGD", 13: "Next Generation Liquefaction",
    14: "Design Safe", 15: "Offshore Wind RVO", 16: "NGTS, Norway",
    17: "Switzerland",
}


def database_group(name):
    """Verified mapping - confirmed against Table 1 with zero unmapped rows."""
    if pd.isna(name):
        return None
    label = str(name).strip()
    if label == "Taiwan_13sites":
        return row_to_database[10]
    if label == "Rhesi":
        return row_to_database[11]
    if label.isdigit():
        n = int(label)
        if n in (2, 3, 4, 5, 6, 7, 8, 9):
            return row_to_database[n]
    match = re.match(r"CPT_db_\d+_(.+)", label)
    if match:
        code = match.group(1).lower()
        code_to_row = {
            "ngl": 13, "designsafe": 14, "offshorewindrvo": 15,
            "kathmandu": 1, "ngts-norway": 16,
        }
        for key, row_num in code_to_row.items():
            if key in code:
                return row_to_database[row_num]
    if "Investigation" in label:
        return row_to_database[12]
    if re.match(r"SCPT7\d+_CPT Data", label):
        return row_to_database[12]
    if "switzerland" in label.lower() or "geoprofile" in label.lower():
        return row_to_database[17]
    return "Other/unmapped"


df["database"] = df[SOURCE_COL].apply(database_group)

unmapped_count = (df["database"] == "Other/unmapped").sum()
if unmapped_count:
    print(f"WARNING: {unmapped_count} rows fell into 'Other/unmapped' - "
          f"check the mapping (e.g. Switzerland label format) before proceeding.")
    print(df.loc[df["database"] == "Other/unmapped", SOURCE_COL].unique()[:20])
else:
    print("Everything mapped cleanly - 0 unmapped rows.")

# ---------------------------------------------------------------------------
# 1b. Add 'country' column, derived from 'database' (same mapping used for
#     Figure 1 / the world map)
# ---------------------------------------------------------------------------
database_to_country = {
    "Nepal": "Nepal",
    "Netherlands (Krogt)": "Netherlands",
    "Snakes": "Netherlands",
    "Baytown, Texas USA": "United States of America",
    "Hollywood, California USA": "United States of America",
    "Emilia Romagna, Italy": "Italy",
    "Terneuzen, Netherlands": "Netherlands",
    "Netherlands (1)": "Netherlands",
    "Premstaller": "Austria",
    "Taiwan": "Taiwan",
    "Rhesi": "Austria",
    "NZGD": "New Zealand",
    "Next Generation Liquefaction": "United States of America",
    "Design Safe": "United States of America",
    "Offshore Wind RVO": "Netherlands",
    "NGTS, Norway": "Norway",
    "Switzerland": "Switzerland",
}

df["country"] = df["database"].map(database_to_country)

unmapped_country_count = df["country"].isna().sum() - df["database"].isna().sum()
if unmapped_country_count > 0:
    print(f"WARNING: {unmapped_country_count} rows have a 'database' value with "
          f"no entry in database_to_country - check the mapping above.")
    missing_dbs = sorted(set(df.loc[df["country"].isna(), "database"].dropna().unique()))
    print("  databases missing a country:", missing_dbs)
else:
    print("Country mapping complete - every mapped database has a country.")

# ---------------------------------------------------------------------------
# 2. Merge SBT columns: prefer sbt_zone_computed, fall back to original SBT
# ---------------------------------------------------------------------------
if "sbt_zone_computed" in df.columns:
    n_filled_by_new = df["sbt_zone_computed"].notna().sum()
    n_filled_by_old_only = (df["sbt_zone_computed"].isna() & df["SBT"].notna()).sum()
    df["SBT"] = df["sbt_zone_computed"].where(
        df["sbt_zone_computed"].notna(), df["SBT"]
    )
    print(f"SBT merge: {n_filled_by_new} rows taken from sbt_zone_computed, "
          f"{n_filled_by_old_only} rows kept from original SBT (computed was missing).")
else:
    print("NOTE: 'sbt_zone_computed' not found - skipping SBT merge.")

# ---------------------------------------------------------------------------
# 3. Drop columns no longer needed in the public release
# ---------------------------------------------------------------------------
cols_to_drop = [c for c in ["qt_mpa", "rf_percent", "sbt_zone_computed", "sbt_name_computed"]
                if c in df.columns]
df = df.drop(columns=cols_to_drop)
print("Dropped columns:", cols_to_drop)

# ---------------------------------------------------------------------------
# 4. Reorder: Dataset, ID, database, country, then everything else
# ---------------------------------------------------------------------------
remaining = [c for c in df.columns if c not in (SOURCE_COL, "ID", "database", "country")]
final_order = [SOURCE_COL, "ID", "database", "country"] + remaining
df = df[final_order]

print("\nFinal columns:", list(df.columns))
print("Final shape:", df.shape)

df.to_csv(OUTPUT, index=False)
print(f"\nSaved to: {OUTPUT}")
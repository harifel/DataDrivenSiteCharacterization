#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Updated 2026-07-28: FOLDER used to be a hardcoded path on another machine
# ("RYZEN 5") that doesn't exist here; switched to relative, matching
# 01_dataset1_choropleth_figure1_v2.py. Also now loads the merged database
# (Dataset 1 + Dataset 3/Geoprofile, Switzerland) as Parquet instead of the
# v3 CSV -- see 15_python scripts/merge_dataset1_dataset3.py.
FOLDER = os.path.join("..", "11_databases")
INPUT = os.path.join(FOLDER, "Dataset1_CPT_database.parquet")


plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
})
#%%

# ---------------------------------------------------------------------------
# 1. Load and map to Table 1 sources
# ---------------------------------------------------------------------------
df = pd.read_parquet(INPUT, columns=["Dataset", "ID", "database", "test type"])
df["ID"] = df["ID"].astype(str)
df = df.dropna(subset=["Dataset"])

# "Netherlands (1)" -> "Netherlands (Vardon)" (merge script relabel). Row 17
# is a *database/source* name like the others, not a country -- must match
# the "database" column's actual value for the merged Dataset 3 rows
# ("Switzerland", anonymized from "Geoprofile"). Using the wrong label here
# would make reindex() silently drop those rows from the heatmap instead of
# counting them.
row_to_database = {
    1: "Nepal", 2: "Netherlands (Krogt)", 3: "Terneuzen, Netherlands",
    4: "Netherlands (Vardon)", 5: "Snakes", 6: "Baytown, Texas USA",
    7: "Hollywood, California USA", 8: "Emilia Romagna, Italy", 9: "Premstaller",
    10: "Taiwan", 11: "Rhesi", 12: "NZGD", 13: "Next Generation Liquefaction",
    14: "Design Safe", 15: "Offshore Wind RVO", 16: "NGTS, Norway", 17: "Switzerland"
}

# Reference numbers matching the manuscript's numbered reference list (kept
# separate from row_to_database above, since that dict's values are also used
# to match the "database" column -- appending "[3]" etc. there would break
# reindex() against the real data). Rhesi and Geoprofile have no reference
# number: both are confidential/unpublished data acknowledged separately in
# the manuscript, not cited sources.
database_to_citation = {
    "Nepal": "3", "Netherlands (Krogt)": "4", "Terneuzen, Netherlands": "5",
    "Netherlands (Vardon)": "6", "Snakes": "7", "Baytown, Texas USA": "8",
    "Hollywood, California USA": "9", "Emilia Romagna, Italy": "10",
    "Premstaller": "11", "Taiwan": "12", "Rhesi": None, "NZGD": "13",
    "Next Generation Liquefaction": "14", "Design Safe": "15",
    "Offshore Wind RVO": "16", "NGTS, Norway": "17", "Switzerland": None,
}


# NEW: warn (don't silently drop) if anything is still unmapped
unmapped_count = (df["database"] == "Other/unmapped").sum()
if unmapped_count:
    print(f"WARNING: {unmapped_count} rows are 'Other/unmapped' and will be "
          f"EXCLUDED from the heatmap by the reindex() step below. Check these labels:")
    print(df.loc[df["database"] == "Other/unmapped", "Dataset"].unique()[:20])
else:
    print("Everything mapped cleanly - 0 unmapped rows.")
#%%
# ---------------------------------------------------------------------------
# 2. Build the CPT/CPTu/SCPT/SCPTu count table
# ---------------------------------------------------------------------------
counts = (
    df.groupby(["database", "test type"])["ID"]
    .nunique()
    .unstack(fill_value=0)
)

row_order = list(row_to_database.values())
col_order = ["CPT", "CPTu", "SCPT", "SCPTu"]
for col in col_order:
    if col not in counts.columns:
        counts[col] = 0
counts = counts.reindex(index=row_order, columns=col_order, fill_value=0)
counts["Total"] = counts.sum(axis=1)
counts.loc["Total"] = counts.sum()

print(counts.to_string())
counts.to_csv(os.path.join(FOLDER, "Table1_FINAL.csv"))
# %%
# ---------------------------------------------------------------------------
# 3. Plot as a heatmap - log-normalized color scale so small counts (9, 25,
#    100...) are still visually distinguishable from each other, instead of
#    all collapsing to near-white under the single 5930 outlier.
# ---------------------------------------------------------------------------
source_rows = counts.drop(index="Total")
total_row = counts.loc[["Total"]]

n_source_rows = len(source_rows)
n_rows_total = n_source_rows + 1

data = np.full((n_rows_total, len(col_order)), np.nan)
data[:n_source_rows, :] = source_rows[col_order].values.astype(float)
data[n_source_rows, :] = total_row[col_order].values.astype(float)
data[data == 0] = np.nan

# Display labels append the citation number (e.g. "Nepal [3]"); the
# underlying source_rows.index values (used for matching/aggregation above)
# are left untouched.
row_labels = [
    f"{name} [{database_to_citation[name]}]" if database_to_citation.get(name) else name
    for name in source_rows.index
] + ["Total"]
row_totals = list(source_rows["Total"]) + [int(total_row["Total"].iloc[0])]

cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(14 * cm, 0.20 * n_rows_total * cm + 3 * cm), dpi=300)

vmin = np.nanmin(data)
vmax = 10000 # np.nanmax(data)
norm = LogNorm(vmin=vmin, vmax=vmax)
im = ax.imshow(data, cmap="Greys", aspect="auto", norm=norm)

ax.set_xticks(range(len(col_order)))
ax.set_xticklabels(col_order, rotation=0, ha="center")
ax.set_yticks(range(n_rows_total))
ax.set_yticklabels(row_labels)
ax.get_yticklabels()[-1].set_fontweight("bold")

# Text color threshold now works on the log scale too: switch to white once
# the log-normalized value crosses ~0.6, so it still tracks visual darkness.
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        if np.isnan(val):
            continue
        is_total_row = (i == n_source_rows)
        color = "white" if norm(val) > 0.6 else "black"
        weight = "bold" if is_total_row else "normal"
        ax.text(j, i, f"{int(val)}", ha="center", va="center", color=color,
                 fontsize=7, fontweight=weight)

for i, total in enumerate(row_totals):
    ax.text(len(col_order) + 0.3, i, f"{int(total)}", ha="left", va="center",
             fontsize=7, fontweight="bold")
ax.text(len(col_order) + 0.3, -1, "Total", ha="left", va="center", fontsize=8, fontweight="bold")

ax.axhline(n_source_rows - 0.5, color="black", linewidth=1.2)

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.15)
cbar.set_label("Number of soundings (log scale)", fontsize=8)
# Explicit ticks at readable round numbers rather than LogNorm's default
# (often crowded/awkward) tick placement.
tick_vals = [v for v in [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000] if vmin <= v <= vmax]
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([str(v) for v in tick_vals])

ax.set_xlim(-0.5, len(col_order) + 1.5)

fig.subplots_adjust(left=0.37, right=0.89, top=0.95, bottom=0.12, wspace=0.05, hspace=0.03)

out_path = os.path.join("Figure2_FINAL_heatmap.png")
plt.savefig(out_path, dpi=1000)
print(f"\nSaved heatmap: {out_path}")
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

# Updated 2026-07-28: switched to a relative FOLDER (matching the other
# Dataset 1 figure scripts) and now loads the merged database (Dataset 1 +
# Dataset 3/Geoprofile, Switzerland) as Parquet instead of the v2 CSV -- see
# 15_python scripts/merge_dataset1_dataset3.py.
FOLDER = os.path.join("..", "11_databases")
INPUT = os.path.join(FOLDER, "Dataset1_CPT_database.parquet")
OUT_BASENAME = os.path.join("Figure5_dataset1_feature_histograms_by_country_grid_new")

# Fixed axis range per feature, shared with 03_plotting_dataset_distribution_figure3.py
# and 04_plotting_distribution2_dataset1_figure4.py so the same feature reads
# the same range across all three figures. 5 evenly-spaced ticks (including
# both endpoints) via np.linspace -- same rule everywhere these limits are used.
AXIS_LIMITS = {
    "Depth (m)": (0, 80),
    "qc (MPa)": (0, 120),
    "fs (MPa)": (0, 4),
    "u2 (MPa)": (0, 4),
    "Vs (m/s)": (0, 800),
}
N_TICKS = 5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COUNTRY_COLORS = cmc.hawaiiS.colors

# ---------------------------------------------------------------------------
# 1. Load. Country comes straight from the "country" column now instead of
# a hand-rolled Dataset-tag -> database -> country lookup (removed
# 2026-07-28): that mapping duplicated what's already in the merged
# database, and had gone stale -- row_to_database said row 8 was
# "Netherlands (Vardon)" but database_to_country's key was still the old
# "Netherlands (1)", so those rows were silently falling into "Unspecified".
# Same fix as 01_dataset1_choropleth_figure1_v2.py.
# ---------------------------------------------------------------------------
df = pd.read_parquet(INPUT, columns=[
    "Dataset", "ID", "country", "Depth (m)", "qc (MPa)", "fs (MPa)", "Rf (%)", "u2 (MPa)", "Vs (m/s)",
])
df["ID"] = df["ID"].astype(str)
df = df.dropna(subset=["Dataset"])
df["country"] = df["country"].fillna("Unspecified")
df["country"] = df["country"].replace({"United States of America": "USA"})

# ---------------------------------------------------------------------------
# 2. Features + physical filters (same as the other Dataset 1 figures)
# ---------------------------------------------------------------------------
features = ["Depth (m)", "qc (MPa)", "fs (MPa)", "u2 (MPa)", "Vs (m/s)"]
short = {
    "Depth (m)": "Depth (m)",
    "qc (MPa)": r"$q_c$ (MPa)",
    "fs (MPa)": r"$f_s$ (MPa)",
    "u2 (MPa)": r"$u_2$ (MPa)",
    "Vs (m/s)": r"$V_s$ (m/s)",
}
for f in features:
    df[f] = pd.to_numeric(df[f], errors="coerce")
df["Rf (%)"] = pd.to_numeric(df["Rf (%)"], errors="coerce")

# ---- Apply CPT physical filters ----
df = df[(df["qc (MPa)"] > 0) & (df["fs (MPa)"] > 0)]
df = df[(df["Rf (%)"] >= 0.001) & (df["Rf (%)"] <= 31)]
df = df[(df["qc (MPa)"] < 200)]
df = df[df["u2 (MPa)"].isna() | ((df["u2 (MPa)"] > 0) & (df["u2 (MPa)"] < 8))]
# df = df[df["Vs (m/s)"].isna() | (df["Vs (m/s)"] < 1500)]
df = df[df["Vs (m/s)"].isna() | ((df["Vs (m/s)"] > 0) & (df["Vs (m/s)"] < 1500))]

# added
df = df[df["Depth (m)"].isna() | (df["Depth (m)"] >= 0)]


country_vals = sorted(df["country"].unique())

# ---------------------------------------------------------------------------
# 3. N x 5 grid: rows = country (vertical), columns = feature (horizontal)
#    plain histograms (raw counts, no density normalization)
# ---------------------------------------------------------------------------
n_rows = len(country_vals)
n_cols = len(features)
cm = 1 / 2.54
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18 * cm, 0.76 * n_rows * cm),
                          dpi=200, sharex="col")

# shared bin edges per column, spanning the fixed AXIS_LIMITS display range
# rather than the raw data min/max -- same fix as
# 04_plotting_distribution2_dataset1_figure4.py. fs/u2/etc. have a long
# outlier tail (fs goes up to ~7.3 MPa even though 95% of values are under
# 0.29 MPa); bins computed from the full range dump almost everything into
# bin 1. Values beyond the axis range are excluded from the histogram now
# (same rows the xlim was already hiding from view).
bin_edges_col = {feat: np.linspace(*AXIS_LIMITS[feat], 26) for feat in features}

# Shared, fixed y-axis (density) range per feature column, instead of every
# panel auto-scaling its own log y-axis independently. With per-panel
# autoscale, a sparse country (e.g. Mexico, N=3 for Vs) has just 1-2
# distinct nonzero density values to anchor the log range against, so it
# stretches to fill the whole panel height regardless of true magnitude --
# making 3 lonely points look like as tall/confident a peak as a country
# with thousands of records. A shared range across all countries for that
# feature fixes that: sparse rows show short bars near the bottom of the
# shared scale, honestly reflecting how little data they have.
ylim_col = {}
for feat in features:
    edges = bin_edges_col[feat]
    bw = edges[1] - edges[0]
    all_nonzero = []
    for country in country_vals:
        vals = df.loc[df["country"] == country, feat].dropna().values
        if len(vals) == 0:
            continue
        counts, _ = np.histogram(vals, bins=edges)
        density = counts / (len(vals) * bw)
        nz = density[density > 0]
        if len(nz):
            all_nonzero.append(nz)
    if all_nonzero:
        all_nonzero = np.concatenate(all_nonzero)
        ylim_col[feat] = (all_nonzero.min() * 0.5, all_nonzero.max() * 2)
    else:
        ylim_col[feat] = None

for i, country in enumerate(country_vals):
    sub = df[df["country"] == country]
    color = 'k'
    for j, feat in enumerate(features):
        ax = axes[i, j]
        edges = bin_edges_col[feat]
        data = sub[feat].dropna().values
        if edges is not None and len(data) > 0:
            # density=True makes peak heights comparable across countries of
            # very different N (Greece has 3.7k records, USA has 2.8M); log
            # y-scale is what actually keeps small tail bars visible next to
            # the dominant low-value peak within the same panel -- density
            # alone doesn't change that ratio, since it's just a per-panel
            # rescale by a constant (N * bin_width).
            ax.hist(data, bins=edges, density=True, color=color,
                    edgecolor="white", linewidth=0.15)
            ax.set_yscale("log")
            if ylim_col[feat] is not None:
                ax.set_ylim(ylim_col[feat])
        ax.set_yticks([])
        ax.set_xlim(AXIS_LIMITS[feat])
        ax.set_xticks(np.linspace(*AXIS_LIMITS[feat], N_TICKS))
        ax.text(0.97, 1.0, f"N={len(data):,}", transform=ax.transAxes,
                ha="right", va="top", color="black", fontsize=6.5)

        # if i == 0:
        #     ax.set_title(short[feat])
        if i == n_rows - 1:
            ax.set_xlabel(short[feat])
        if j == 0:
            ax.set_ylabel(country, rotation=0, ha="right", va="center",
                          labelpad=8)
            


fig.subplots_adjust(left=0.14, right=0.97, top=0.99, bottom=0.09, wspace=0.2, hspace=0.2)

fig.savefig(f"{OUT_BASENAME}.png", dpi=1000)
print(f"Saved {OUT_BASENAME}.png")
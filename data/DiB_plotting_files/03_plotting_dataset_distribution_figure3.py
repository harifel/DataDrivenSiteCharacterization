import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Updated 2026-07-28: FOLDER used to be a hardcoded path on another machine
# ("RYZEN 5") that doesn't exist here; switched to relative, matching
# 01_dataset1_choropleth_figure1_v2.py / 02_plotting_dataset1_Figure2_v2.py.
# Also now loads the merged database (Dataset 1 + Dataset 3/Geoprofile,
# Switzerland) as Parquet instead of the v3 CSV -- see
# 15_python scripts/merge_dataset1_dataset3.py.
FOLDER = os.path.join("..", "11_databases")
INPUT = os.path.join(FOLDER, "Dataset1_CPT_database.parquet")
OUT_BASENAME = os.path.join("Figure3_dataset1_scatter_matrix_new")

# Fixed axis range per feature, shared with 04_plotting_distribution2_dataset1_figure4.py
# and 05_plotting_distribution_dataset1_figure5.py so the same feature reads
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

# 1. Load Dataset
# "Rf (%)" isn't in `features` (not plotted directly) but is read here anyway
# since it's used below as one of the outlier-filtering criteria.
df = pd.read_parquet(INPUT, columns=[
    "Dataset", "ID", "Depth (m)", "qc (MPa)", "fs (MPa)", "Rf (%)", "u2 (MPa)", "Vs (m/s)",
])
df["ID"] = df["ID"].astype(str)
df = df.dropna(subset=["Dataset"])

# 2. Features for the 6x6 matrix
features = ["Depth (m)", "qc (MPa)", "fs (MPa)", "u2 (MPa)", "Vs (m/s)"]
short = {
    "Depth (m)": "Depth (m)", 
    "qc (MPa)": "q$_c$ (MPa)", 
    "fs (MPa)": "f$_s$ (MPa)",
    "u2 (MPa)": "u$_2$ (MPa)",
    # "Rf (%)": "R$_f$ (%)",     # FIXED: Removed trailing colon typo
    "Vs (m/s)": "V$_s$ (m/s)",
}

# Parquet preserves real dtypes (unlike CSV), so these are already float64
# coming out of read_parquet -- no per-column pd.to_numeric coercion needed.

# ---- Apply CPT physical filters ----
df = df[(df["qc (MPa)"] > 0) & (df["fs (MPa)"] > 0)]
df = df[(df["Rf (%)"] >= 0.001) & (df["Rf (%)"] <= 31)]
df = df[(df["qc (MPa)"] < 200)]
df = df[df["u2 (MPa)"].isna() | ((df["u2 (MPa)"] > 0) & (df["u2 (MPa)"] < 8))]
# df = df[df["Vs (m/s)"].isna() | (df["Vs (m/s)"] < 1500)]
df = df[df["Vs (m/s)"].isna() | ((df["Vs (m/s)"] > 0) & (df["Vs (m/s)"] < 1500))]

# added
df = df[df["Depth (m)"].isna() | (df["Depth (m)"] >= 0)]


# ------------------------------------
# ------------------------------------

n = len(features)
cm = 1 / 2.54
fig, axes = plt.subplots(n, n, figsize=(18 * cm, 18 * cm), dpi=200)

for i in range(n):          # row
    for j in range(n):      # col
        ax = axes[i, j]
        yf, xf = features[i], features[j]

        # Valid mutually available data mask
        mask = df[xf].notna() & df[yf].notna()
        valid_x = df.loc[mask, xf]
        valid_y = df.loc[mask, yf]
        num_pts = len(valid_x)

        if i == j:
            # ---- Diagonal: Distribution histogram ----
            # Bins span the fixed AXIS_LIMITS range, not this panel's own
            # data min/max -- fs/u2/etc. have a long outlier tail (fs goes up
            # to ~7.3 MPa even though 95% of values are under 0.29 MPa), and
            # bins computed from bins=30 over the full range would dump
            # almost everything into bin 1. Values beyond the axis range are
            # excluded from the histogram (same rows the xlim already hides).
            # density=True + log y-scale for the same reason as Figures 4/5:
            # keeps the small tail bars visible next to the dominant
            # low-value peak. No shared/fixed ylim needed here though --
            # unlike Figures 4/5 there's only one panel per feature (the
            # whole dataset, not one row per SBT class/country), so there's
            # nothing to compare across that a per-panel autoscale would
            # distort.
            data = df[xf].dropna().values
            ax.hist(data, bins=np.linspace(*AXIS_LIMITS[xf], 31), density=True,
                     color="k", edgecolor="white", linewidth=0.2)
            ax.set_yscale("log")
            # set_yticks([]) alone only clears major ticks -- log scale keeps
            # auto-generating minor tick marks regardless, so those need to
            # be turned off explicitly too.
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
            ax.set_xlim(AXIS_LIMITS[xf])
            ax.set_xticks(np.linspace(*AXIS_LIMITS[xf], N_TICKS))

        elif j > i:
            # ---- Upper Diagonal: Pearson r and Sample Size ----
            ax.set_xticks([])
            ax.set_yticks([])

            if num_pts > 1:
                r_val, _ = pearsonr(valid_x, valid_y)
                text_str = f"r = {r_val:.2f}\nN = {num_pts:,}"
            else:
                text_str = f"N = {num_pts:,}"

            ax.text(0.5, 0.5, text_str, transform=ax.transAxes,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=8, fontweight='bold', color='k')

        else:
            # ---- Lower Diagonal: Scatter plot ----
            # plot(marker=".") instead of scatter(): scatter() builds a
            # per-point PathCollection, which is inherently slow once panels
            # have millions of points (some pairs here run up to ~12.5M).
            # plot() with a single shared marker style renders as one Line2D
            # artist instead, which is substantially faster at this scale.
            # rasterized=True keeps it fast/small if this ever gets exported
            # to a vector format (PDF/SVG) later, though the final output
            # here is PNG anyway. Same points, same alpha/size/color as
            # before -- purely a rendering-speed change, nothing visual.
            ax.plot(valid_x, valid_y, marker=".", markersize=2.8, alpha=0.5,
                     color="k", markeredgewidth=0, linestyle="none", rasterized=True)
            ax.set_xlim(AXIS_LIMITS[xf])
            ax.set_ylim(AXIS_LIMITS[yf])
            ax.set_xticks(np.linspace(*AXIS_LIMITS[xf], N_TICKS))
            ax.set_yticks(np.linspace(*AXIS_LIMITS[yf], N_TICKS))

        # ---- X-axis formatting (Only labels on the bottom row) ----
        if i == n - 1:
            ax.set_xlabel(short[xf])
        else:
            ax.set_xticklabels([])

        # ---- Y-axis formatting (Only labels on the first column) ----
        if j == 0:
            ax.set_ylabel(short[yf])
        else:
            ax.set_yticklabels([])

# Tight spacing adjustments
fig.subplots_adjust(left=0.08, right=0.95, top=0.99, bottom=0.07, wspace=0.2, hspace=0.2)

fig.savefig(f"{OUT_BASENAME}.png", dpi=1000)
print(f"Saved {OUT_BASENAME}.png")
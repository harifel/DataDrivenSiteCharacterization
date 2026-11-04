"""
dataset2_input_output_sbt.py

Figures 6a/6b/6c for the Data Description section: every coupled lab/index
parameter ("output") compared against the three CPT inputs Depth, qc, and
fs ("input"), grouped by Robertson (2010) SBT zone, plus a 4th column
showing how many records that parameter has in each SBT zone.

Layout: fixed 3 scatter columns (Depth, qc, fs) so a reader can scan across
one row and see how a single lab parameter varies against all three CPT
inputs at once, plus a 4th column with a simple categorical bar chart:
one bar per SBT zone, height = record count for that (parameter, zone)
pair. This is a DATA AVAILABILITY chart, not a value distribution — it
answers "how much data do we have per zone for this parameter", not "what
values does this parameter take." Same SBT color scheme as the scatter
panels. One row per lab parameter. All 21 parameters is too dense for one
figure, so this is split into three figures of 7 parameters each, in the
same order used throughout (state/index -> shear-strength framework ->
undrained strength & stiffness), sharing one SBT color scheme and legend
so the three figures read as one consistent set.

Depth is kept linear-x (a physically additive quantity); qc/fs are also
linear here as configured, with x-axes starting at 0.

The count column uses the full set of non-missing (parameter, SBT) pairs
for that row, independent of which x-columns happen to be missing for a
given record — so its total n can be slightly higher than any individual
scatter panel's n in the same row.

Colors use cmcrameri's batlowS — the categorical/qualitative variant of
the batlow scientific colormap (perceptually uniform, colorblind-safe).

SBT zone 0 (16 records) is not a standard Robertson (2010) zone number and
is shown as "0 / unclassified" — verify against your source data whether
this is a genuine "no classification" flag or something else.

Parameter columns with zero non-missing records, and E50ref_m_parameter,
are dropped — same convention as the other Dataset 2 figures.

Output: dataset2_input_output_sbt_part1.png / _part2.png / _part3.png,
each 19 cm wide (Elsevier double-column), height scaled to 7 rows.

Update INPUT_XLSX below if the file moves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmcrameri.cm as cmc
import pyarrow.parquet as pq

cm = 1 / 2.54

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize":  9,
    "axes.grid":       False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 9,
})


# ---- user-updatable paths -------------------------------------------------
# Updated 2026-07-28: loads the Parquet conversion of the withSBT workbook
# instead of the xlsx directly -- see 15_python scripts/convert_dataset2_to_parquet.py.
INPUT_PARQUET = "..\\11_databases\\Dataset2_Coupled_Geotechnical_CPT.parquet"
OUT_BASENAME = "Figure9_dataset2_input_output_sbt"
# ----------------------------------------------------------------------------

# --- same lab parameter list/order/drop-rule as the other Dataset 2 figures
param_cols = [
    "gamma (kN/m3)", "gamma_d (kN/m3)", "water_content (%)",
    "e0 (-)", "OCR (-)",
    "phi_CD (degree)", "cohesion_CD (kPa)",
    "phi_RS_peak (degree)", "phi_RS_residual (degree)",
    "phi_SB (degree)", "cohesion_SB (kPa)",
    "su_CU_remoulded (kPa)", "su_CU_undisturbed (kPa)",
    "su_CUe_undisturbed (kPa)", "su_DSS (kPa)",
    "su_UU_remoulded (kPa)", "su_UU_undisturbed (kPa)", "su_FCT (kPa)",
    "torvane (kPa)", "pocket_penetrometer (kPa)",
    "E50ref (MPa)", "E50ref_m_parameter (-)", "Eoed (MPa)",
]
# "su_CU_remoulded (kPa)" has never actually existed in this workbook (only
# su_CU_undisturbed / su_UU_remoulded do) -- pd.read_excel used to silently
# tolerate that; pd.read_parquet's columns= is strict, so filter against the
# real schema first (schema-only, no data read).
existing_cols = set(pq.ParquetFile(INPUT_PARQUET).schema.names)
param_cols = [c for c in param_cols if c in existing_cols]

df = pd.read_parquet(INPUT_PARQUET, columns=param_cols + ["Depth (m)", "qc (MPa)", "fs (MPa)", "Rf (%)", "u2 (MPa)", "SBT"])
df = df.replace(r"^\s*$", np.nan, regex=True)

df = df[(df["qc (MPa)"] > 0) & (df["fs (MPa)"] > 0)]
df = df[(df["Rf (%)"] >= 0.001) & (df["Rf (%)"] <= 31)]
df = df[(df["qc (MPa)"] < 200)]
df = df[df["u2 (MPa)"].isna() | ((df["u2 (MPa)"] > 0) & (df["u2 (MPa)"] < 8))]


param_cols = [c for c in param_cols if df[c].notna().sum() > 0]
param_cols = [c for c in param_cols if c != "E50ref_m_parameter (-)"]

short_label = {
    "gamma (kN/m3)": r"$\gamma_{sat}$ (kN/m$^3$)",
    "gamma_d (kN/m3)": r"$\gamma_d$ (kN/m$^3$)",
    "water_content (%)": r"$w_n$ (%)",
    "e0 (-)": r"$e_0$ (-)",
    "OCR (-)": r"$OCR$ (-)",
    "phi_CD (degree)": r"$\varphi'_{CD}$ ($^{\circ}$)",
    "cohesion_CD (kPa)": r"$c'_{CD}$ (kPa)",
    "phi_RS_peak (degree)": r"$\varphi_{RS,peak}$ ($^{\circ}$)",
    "phi_RS_residual (degree)": r"$\varphi_{RS,res}$ ($^{\circ}$)",
    "phi_SB (degree)": r"$\varphi_{SB}$ ($^{\circ}$)",
    "cohesion_SB (kPa)": r"$c_{SB}$ (kPa)",
    "su_CU_remoulded (kPa)": r"$s_{u,CU,rem}$ (kPa)",
    "su_CU_undisturbed (kPa)": r"$s_{u,CU}$ (kPa)",
    "su_CUe_undisturbed (kPa)": r"$s_{u,CUe}$ (kPa)",
    "su_DSS (kPa)": r"$s_{u,DSS}$ (kPa)",
    "su_UU_remoulded (kPa)": r"$s_{u,UU,rem}$ (kPa)",
    "su_UU_undisturbed (kPa)": r"$s_{u,UU}$ (kPa)",
    "su_FCT (kPa)": r"$s_{u,FCT}$ (kPa)",
    "torvane (kPa)": r"$s_{u,torvane}$ (kPa)",
    "pocket_penetrometer (kPa)": r"$s_{u,pocket}$ (kPa)",
    "E50ref (MPa)": r"$E_{50}^{ref}$ (MPa)",
    "Eoed (MPa)": r"$E_{oed}$ (MPa)",
}

# split into 3 chunks of 7 (21 params exactly -> no leftover)
CHUNK_SIZE = 7
chunks = [param_cols[i:i + CHUNK_SIZE] for i in range(0, len(param_cols), CHUNK_SIZE)]

# --- SBT zone styling (shared across all 3 figures) -------------------------
sbt_names = {
    0: "0: unclassified",
    1: "1: Sensitive, fine grained",
    2: "2: Clay - organic soil",
    3: "3: Clays: clay to silty clay",
    4: "4: Silt mixtures: clayey silt to silty clay",
    5: "5: Sand mixtures: silty sand to sandy silt",
    6: "6: Sands: clean sand to silty sand",
    7: "7: Dense sand to gravelly sand",
    8: "8: Stiff sand to clayey sand*",
    9: "9: Stiff fine-grained*",
}
batlowS_colors = cmc.batlowS.colors[:10]
zone_style = {z: {"color": batlowS_colors[z], "marker": "."} for z in range(10)}

zones_present = sorted(pd.to_numeric(df["SBT"], errors="coerce").dropna().astype(int).unique())
zone_counts = pd.to_numeric(df["SBT"], errors="coerce").dropna().astype(int).value_counts()

INPUTS = [
    ("Depth (m)", r"Depth (m)", False),
    ("qc (MPa)", r"$q_c$ (MPa)", True),
    ("fs (MPa)", r"$f_s$ (MPa)", True),
]


def scatter_by_zone(ax, x, y, z):
    for zone in zones_present:
        m = z == zone
        if m.sum() == 0:
            continue
        style = zone_style[zone]
        ax.scatter(x[m], y[m], s=50, alpha=0.75, color=style["color"],
                    marker='.', linewidths=0, zorder=3)


def count_by_zone_bar(ax, z):
    """Simple categorical bar chart: one bar per SBT zone, height = number
    of records this parameter has in that zone. Data availability, not a
    value distribution — deliberately not binned by the parameter's own
    value range."""
    counts = pd.Series(z).value_counts()
    xs = list(range(len(zones_present)))
    heights = [counts.get(zone, 0) for zone in zones_present]
    colors = [zone_style[zone]["color"] for zone in zones_present]
    ax.bar(xs, heights, color=colors, edgecolor="none", zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(z) for z in zones_present], fontsize=5)
    ax.set_ylabel("count")


legend_handles = [
    Line2D([0], [0], marker=zone_style[z]["marker"], color="none",
           markerfacecolor=zone_style[z]["color"], markeredgewidth=0,
           markersize=10, label=f"{sbt_names.get(z, z)} (N={zone_counts.get(z, 0)})")
    for z in zones_present
]

for part_idx, chunk in enumerate(chunks, start=1):
    n_rows = len(chunk)
    n_cols = len(INPUTS) + 1  # +1 for the stacked-histogram column

    fig_w = 19 * cm
    fig_h = (2.75 * n_rows) * cm
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, dpi = 500)

    for row_idx, col in enumerate(chunk):
        # decide once per row whether this parameter's own values need log-y,
        # same rule used by the scatter panels below, reused for the histogram
        y_full = pd.to_numeric(df[col], errors="coerce").dropna()
        row_log_y = bool(len(y_full) and y_full.min() > 0 and y_full.max() / y_full.min() > 50)

        for col_idx, (xcol, xlabel, xlog) in enumerate(INPUTS):
            ax = axes[row_idx, col_idx]
            sub = df[[xcol, col, "SBT"]].apply(pd.to_numeric, errors="coerce").dropna()
            sub["SBT"] = sub["SBT"].astype(int)
            x, y, z = sub[xcol].values, sub[col].values, sub["SBT"].values

            scatter_by_zone(ax, x, y, z)

            if row_log_y:
                ax.set_yscale("log")

            if row_idx == 0:
                ax.set_title(xlabel, pad=2)
            if col_idx == 0:
                ax.set_ylabel(short_label.get(col, col))
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel)
            ax.text(0.97, 0.4, f"N={len(sub)}", transform=ax.transAxes,
                     ha="right", va="bottom", color="#555555")
            # E50ref's y-values are all referenced to p_ref=100 kPa; noted
            # here (not in the y-label) since that label is rotated and this
            # row is too short to fit the longer subscripted label text.
            if col == "E50ref (MPa)":
                ax.text(0.3, 0.90, r"$p_{ref}=100$ kPa", transform=ax.transAxes,
                         ha="left", va="bottom", fontsize=6, color="#555555",
                         style="italic")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            ax.set_xlim(xmin=0)

        # --- 4th column: record count per SBT zone for this parameter -----
        count_ax = axes[row_idx, len(INPUTS)]
        hsub = df[[col, "SBT"]].apply(pd.to_numeric, errors="coerce").dropna()
        hsub["SBT"] = hsub["SBT"].astype(int)
        count_by_zone_bar(count_ax, hsub["SBT"].values)
        if row_idx == 0:
            count_ax.set_title("records per SBT zone", pad=2)
        if row_idx == n_rows - 1:
            count_ax.set_xlabel("SBT zone")
        count_ax.text(0.97, 0.92, f"N={len(hsub)}", transform=count_ax.transAxes,
                       ha="right", va="top", color="#555555")
        for spine in ("top", "right"):
            count_ax.spines[spine].set_visible(False)

    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.01), frameon=False,
                title="SBT zone (Robertson, 2010)")

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.97,
        bottom=0.2,
        wspace=0.4,
        hspace=0.5
    )

    out_path = f"{OUT_BASENAME}_part{part_idx}.png"
    fig.savefig(out_path, dpi=1000)
    print(f"Saved {out_path} ({n_rows} rows x {n_cols} cols): {chunk}")
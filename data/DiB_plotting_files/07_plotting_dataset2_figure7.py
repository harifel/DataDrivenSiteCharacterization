"""
dataset2_availability_heatmap.py

Generates a data-availability heatmap for Dataset 2 (Coupled Geotechnical and
CPT Data): rows = site, columns = laboratory/index parameter, cell value =
number of records at that site with a non-missing value for that parameter.

Depth/qc/fs/Rf/u2/SBT/Destination/CPT_name are CPT-stream or metadata columns
and are intentionally excluded from the heatmap — it is meant to show which
sites contribute which *coupled laboratory* parameters, which is the whole
point of Dataset 2.

Parameter columns that are entirely empty across all sites (e.g.
su_CU_remoulded, which has zero non-missing records) are dropped
automatically before plotting, rather than hardcoded — so the figure stays
correct if the underlying data changes.

Output: dataset2_availability_heatmap.png, sized to Elsevier's double-column
width (19 cm).

Update INPUT_XLSX below if the file moves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm

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
# INPUT_PARQUET = "..\\11_databases\\Dataset2_Coupled_Geotechnical_CPT_20260728.parquet"
INPUT_PARQUET = "..\\11_databases\\Dataset2_Coupled_Geotechnical_CPT.parquet"
OUT_BASENAME = "Figure7_dataset2_availability_heatmap"
# ----------------------------------------------------------------------------

param_cols_all = [
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
# tolerate that via the "if c in df.columns" filter below, but
# pd.read_parquet's columns= is strict and errors on an unknown name, so
# filter against the real schema first (schema-only, no data read).
existing_cols = set(pq.ParquetFile(INPUT_PARQUET).schema.names)
param_cols_all = [c for c in param_cols_all if c in existing_cols]
df = pd.read_parquet(INPUT_PARQUET, columns=["sampling_name (-)"] + param_cols_all)

# Clean stray whitespace-only strings (e.g. Eoed contains 20 cells of " ")
# so they are correctly treated as missing rather than as text.
df = df.replace(r"^\s*$", np.nan, regex=True)


# --- group individual boreholes/samples into the 16 reporting sites --------
def site_group(name):
    if pd.isna(name):
        return None
    n = str(name).strip()
    mapping = [
        ("Austria_site", "Austria (private)"),
        ("Ballina", "Australia [18]"),
        ("Bergambacht", "Bergambacht, NL [19]"),
        ("DOV", "DOV, NL [19]"),
        ("Delft", "Delft, NL [19]"),
        ("Full Scale Damwand Proef Eemdijk", "D.P. Eemdijk, NL [19]"),
        ("HKW", "Offshore Wind RVO, NL [16]"),
        ("Kinderdijk", "Kinderdijk, NL [19]"),
        ("Koehool", "Koehool, NL [19]"),
        ("MMD Katwoude", "MMD Katwoude, NL [19]"),
        ("MMD Pieterman", "MMD Pieterman, NL [19]"),
        ("NGTS", "NGTS, Norway [17]"),
        ("NWZ", "Offshore Wind RVO, NL [16]"),
        ("Norway_site", "Norway (private)"),
        ("PITS_Rhesi", "Austria (private)"),
        ("PITS_", "Austria (private)"),
        ("PTS_", "Austria (private)"),
        ("Salzburg", "Austria (private)"),
        ("Schalkwijk", "Schalkwijk, NL [19]"),
        ("Streefkerk", "Streefkerk, NL [19]"),
        ("TNW", "Offshore Wind RVO, NL [16]"),
        ("Waardenburg", "Waardenburg, NL [19]"),
    ]
    for prefix, group in mapping:
        if n.startswith(prefix):
            return group
    return "Other/unmapped"


df["site"] = df["sampling_name (-)"].apply(site_group)

# --- parameters to show (coupled lab / index parameters only) --------------
param_cols = [c for c in param_cols_all if c in df.columns]

# shorter labels for the plot, in matplotlib mathtext (LaTeX-style) with units
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
    "pocket_penetrometer (kPa)": r"$s_{u,pocket}$(kPa)",
    "E50ref (MPa)": r"$E_{50}^{ref}$ (MPa)",
    "E50ref_m_parameter (-)": r"$m$ (-)",
    "Eoed (MPa)": r"$E_{oed}$ (MPa)",
}

# Reference numbers here must match the group labels returned by
# site_group() above; sorted alphabetically (case-insensitive) so the order
# doesn't depend on where a site happens to sit in that mapping.
site_order = [
    "Australia [18]", "Austria (private)",
    "Bergambacht, NL [19]", "D.P. Eemdijk, NL [19]", "Delft, NL [19]",
    "DOV, NL [19]", "Kinderdijk, NL [19]", "Koehool, NL [19]",
    "MMD Katwoude, NL [19]", "MMD Pieterman, NL [19]", "NGTS, Norway [17]",
    "Norway (private)", "Offshore Wind RVO, NL [16]", "Schalkwijk, NL [19]",
    "Streefkerk, NL [19]", "Waardenburg, NL [19]",
]
site_order = [s for s in site_order if s in df["site"].unique()]

counts = (
    df[df["site"].isin(site_order)]
    .groupby("site")[param_cols]
    .apply(lambda g: g.notna().sum())
    .reindex(site_order)
)

# --- drop parameter columns with zero records across every site ------------
empty_cols = counts.columns[counts.sum(axis=0) == 0].tolist()
if empty_cols:
    print("Dropping empty columns (0 records across all sites):", empty_cols)
    
empty_cols.append("E50ref_m_parameter (-)")

counts = counts.drop(columns=empty_cols)
param_cols = [c for c in param_cols if c not in empty_cols]



matrix = counts.values.astype(float)
matrix[matrix == 0] = np.nan  # blank cells for zero-availability

# --- append a Total row (per parameter) and Total column (per site) --------
# These carry the row/column sums plus a grand total in the corner. They are
# excluded from the colour mapping (kept white) and drawn in bold, separated
# from the count cells by a thicker border, so they read as a summary rather
# than as another data cell.
col_totals = counts.sum(axis=0).values  # per parameter, across sites
row_totals = counts.sum(axis=1).values  # per site, across parameters
grand_total = counts.values.sum()

n_rows, n_cols = matrix.shape
matrix_ext = np.full((n_rows + 1, n_cols + 1), np.nan)
matrix_ext[:n_rows, :n_cols] = matrix
# color-map is only ever applied to the original data block (matrix_ext left
# NaN elsewhere), so the totals stay uncoloured regardless of the cmap used.

row_labels = site_order + ["Total"]
col_labels = [short_label.get(c, c) for c in param_cols] + ["Total"]

# ---- plot -------------------------------------------------------------
fig_w = 19 * cm  # Elsevier double-column width
fig_h = 7.4 * cm
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=500)

# Greyscale colormap: light = few records, dark = many records.
cmap = plt.cm.Greys
cmap = cmap.copy()
cmap.set_bad(color="white")

# Log-scale color mapping: availability counts span roughly 1 to a few
# hundred, so a linear scale makes the low-count cells (which are most of
# the figure) all look the same near-white shade. vmin=1 since cells with
# 0 records are already NaN'd out above (log(0) is undefined anyway).
vmax = np.nanmax(matrix)
norm = LogNorm(vmin=1, vmax=vmax)
im = ax.imshow(matrix_ext, cmap=cmap, aspect="auto", norm=norm)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=90)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)

# annotate counts — text colour follows the actual cmap colour at that
# value (via the same LogNorm used for the image), not a fixed threshold,
# so it stays legible on every cell.
for i in range(n_rows):
    for j in range(n_cols):
        v = matrix_ext[i, j]
        if not np.isnan(v):
            frac = norm(v)
            txt_color = "white" if frac > 0.55 else "black"
            ax.text(j, i, int(v), ha="center", va="center",
                     color=txt_color, fontsize=8)

# Total column (per site) and Total row (per parameter), plus grand total —
# uncoloured cells, bold black text.
for i, v in enumerate(row_totals):
    ax.text(n_cols, i, int(v), ha="center", va="center",
             color="black", fontweight="bold", fontsize=7)
for j, v in enumerate(col_totals):
    ax.text(j, n_rows, int(v), ha="center", va="center",
             color="black", fontweight="bold", fontsize=7)
ax.text(n_cols, n_rows, int(grand_total), ha="center", va="center",
         color="black", fontweight="bold", fontsize=7)

ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
ax.grid(which="minor", color="k", linewidth=0.6)
ax.tick_params(which="minor", bottom=False, left=False)

# thicker border separating the Total row/column from the count cells
ax.axhline(n_rows - 0.5, color="k", linewidth=1.4)
ax.axvline(n_cols - 0.5, color="k", linewidth=1.4)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Records available")

fig.subplots_adjust(
    left=0.23,
    right=0.94,
    top=0.99,
    bottom=0.33,
    wspace=0.05,
    hspace=0.03
)

plt.show()
fig.savefig(f"{OUT_BASENAME}.png", dpi=1000)
print(counts.sum(axis=1).rename("total non-missing cells per site"))
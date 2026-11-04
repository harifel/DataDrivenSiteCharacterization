"""
dataset2_parameter_raincloud.py

Figure 3 (alternative) for the Data Description section: for every coupled
lab/index parameter, shows the individual data points (jittered scatter,
so every record is visible — nothing hidden inside a box/whisker summary)
together with a histogram of the same values on the same axis ("raincloud"
style: cloud of points + rain of the histogram).

Each panel's x-axis is labelled to make the two halves self-explanatory:
the scatter half reads "records" (every point = one measurement), and the
histogram half is given its own 0-to-max-bin-count scale, so bar length
has a concrete numeric reference rather than just a visual impression.

This is more transparent than a plain box plot: box plots summarise away
exactly the kind of thing that matters here (e.g. the peat-related outliers
in e0/w_n/gamma_sat cluster visibly as a separate group of points rather
than disappearing into "outlier dots").

Parameter columns with zero non-missing records are dropped automatically.

Output: dataset2_parameter_raincloud.png, square 19 x 19 cm.

Update INPUT_XLSX below if the file moves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

cm = 1 / 2.54
rng = np.random.default_rng(42)  # fixed seed -> reproducible jitter

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
OUT_BASENAME = "Figure8_dataset2_parameter_raincloud"
# ----------------------------------------------------------------------------

# --- parameters to show (coupled lab / index parameters only) --------------
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

df = pd.read_parquet(INPUT_PARQUET, columns=param_cols + ["Depth (m)", "qc (MPa)", "fs (MPa)", "Rf (%)", "u2 (MPa)", "SBT", "Horizontal distance (m)"])
df = df.replace(r"^\s*$", np.nan, regex=True)

# Existing filters
df = df[(df["qc (MPa)"] > 0) & (df["fs (MPa)"] > 0)]
df = df[(df["Rf (%)"] >= 0.001) & (df["Rf (%)"] <= 31)]
df = df[(df["qc (MPa)"] < 200)]
df = df[df["u2 (MPa)"].isna() | ((df["u2 (MPa)"] > 0) & (df["u2 (MPa)"] < 8))]

# df = df[df["water_content (%)"] < 100]
# df = df[df["gamma (kN/m3)"] > 10]
# df = df[df["OCR (-)"] < 10]
# df = df[df["e0 (-)"] < 2]

# # --- Handle all 'su' columns ---
# su_cols = [col for col in df.columns if "su" in col]

# # Keeps rows where all non-null 'su' values are < 300
# df = df[df[su_cols].apply(lambda row: all(val < 300 for val in row if pd.notna(val)), axis=1)]



param_cols = [c for c in param_cols if df[c].notna().sum() > 0]

# m (E50ref_m_parameter) dropped from the figure — not a physically comparable
# quantity alongside the others and only clutters the grid.
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
    "pocket_penetrometer (kPa)": r"$s_{u,pocket} $(kPa)",
    "E50ref (MPa)": r"$E_{50, p_{ref} = 100~kPa}^{ref}$ (MPa) ",
    "E50ref_m_parameter (-)": r"$m$ (-)",
    "Eoed (MPa)": r"$E_{oed}$ (MPa)",
}


def rainpanel(ax, data, label):
    """One parameter: jittered points (left) + histogram (right), shared
    y-axis (the parameter's actual values)."""
    n = len(data)
    if n == 0:
        ax.axis("off")
        return

    # histogram, drawn as horizontal bars growing rightward from x=0.05
    n_bins = int(np.clip(np.sqrt(n), 5, 16))
    counts, edges = np.histogram(data, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    max_count = counts.max() if counts.max() > 0 else 1
    hist_span = 0.38
    hist_left = 0.06
    ax.barh(centers, counts / max_count * hist_span, height=width * 0.9,
             left=hist_left, color="#bfbfbf", edgecolor="#8c8c8c", linewidth=0.3,
             align="center", zorder=2)

    # jittered raw points, to the left of x=0
    pts_lo, pts_hi = -0.34, -0.06
    jitter = rng.uniform(pts_lo, pts_hi, size=n)
    ax.scatter(jitter, data, s=3, color="#2b2b2b", alpha=0.35,
                linewidths=0, zorder=3)

    ax.set_xlim(-0.4, 0.48)
    # x-axis identifies what each half actually shows: the scatter half is
    # labelled "records" (every individual data point); the histogram half
    # gets its own count scale (0 to the tallest bin) so bar length has a
    # numeric reference instead of just "a feeling" for the shape.
    pts_mid = (pts_lo + pts_hi) / 2
    ax.set_xticks([pts_mid, hist_left, hist_left + hist_span])
    ax.set_xticklabels(["records", "0", str(int(max_count))],)
    ax.tick_params(axis="x", length=2, pad=1)

    ax.set_title(label, pad=2)
    ax.text(0.98, 0.3, f"N={n}", transform=ax.transAxes,
             ha="right", va="bottom", color="#555555")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---- figure: square 19 x 19 cm grid --------------------------------------
n_params = len(param_cols)

# Pick n_cols so the grid fills as evenly as possible — a nearly-square
# panel count with no (or minimal) leftover in the last row. For 21
# parameters this lands on 7x3, filling completely with no orphan row
# (5x5 would leave a single stranded panel, e.g. Eoed alone on row 5).
best = None
for cand_cols in range(3, n_params + 1):
    cand_rows = int(np.ceil(n_params / cand_cols))
    leftover = cand_rows * cand_cols - n_params
    aspect_penalty = abs(cand_cols - cand_rows)  # prefer closer to square
    score = (leftover, aspect_penalty)
    if best is None or score < best[0]:
        best = (score, cand_cols, cand_rows)
n_cols, n_rows = best[1], best[2]

fig_w = 19 * cm
fig_h = 19 * cm
fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi = 1000)
axes = np.atleast_2d(axes)

for idx, col in enumerate(param_cols):
    r, c = divmod(idx, n_cols)
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    rainpanel(axes[r, c], data, short_label.get(col, col))

for idx in range(n_params, n_rows * n_cols):
    r, c = divmod(idx, n_cols)
    axes[r, c].axis("off")

fig.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.8)
fig.savefig(f"{OUT_BASENAME}.png", dpi=1000)
print(f"Saved {OUT_BASENAME}.png ({n_rows}x{n_cols} panels, {n_params} parameters)")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Simple 3x2 figure (left column = su, right column = Vs):
#   row 1 = histograms of undrained shear strength su (Dataset 2, pooled
#           across lab test types: su_CU_undisturbed, su_CUe_undisturbed,
#           su_DSS, su_UU_remoulded, su_UU_undisturbed, su_FCT -- index-test
#           proxies torvane/pocket_penetrometer are excluded, not true lab
#           su) and Vs (Dataset 1).
#   row 2 = histograms of the depth of those same records (depth of the su
#           measurements, depth of the Vs measurements).
#   row 3 = smoothed 2D density of the same two parameters against depth
#           (depth on the y-axis, increasing downward): a fine-grained 2D
#           histogram passed through a Gaussian filter and rendered with
#           smooth shading, so the concentration of records at shallow
#           depth -- and the sparse tail at depth -- reads as a continuous
#           density surface rather than blocky bins or an overplotted
#           scatter. This is the depth bias in the data: few deep records
#           means model predictions at depth rest on much less evidence.
#
# Produces TWO figures: a plain version, and a version with literature
# reference values (from the DGGT su/G0 case-study figure) overlaid on the
# row-3 density panels, so the database's depth bias can be judged directly
# against real, independently-measured points.
#   - su reference points (CIUC, CAUC, Vane83, VaneGSC, Vane72) are used
#     as-is (same units, su vs depth).
#   - the G0 reference points (Conetec CPT02-2012, Conetec SCPT14-01,
#     Yafrate & Dejong 2006, Landon 2007, Styler & Mayne 2013) are converted
#     to Vs via Vs = sqrt(G0 / rho), using the site's own measured unit
#     weight profile (tube samples, "Unit"/"z_unit") interpolated to each
#     G0 point's depth -- not an assumed constant density.
FOLDER = os.path.join("..", "11_databases")
INPUT_DATASET1 = os.path.join(FOLDER, "Dataset1_CPT_database.parquet")
INPUT_DATASET2 = os.path.join(FOLDER, "Dataset2_Coupled_Geotechnical_CPT_20260730_withSBT.parquet")
OUT_BASENAME = "Figure10_vs_su_distributions"

REF_EXCEL = r"C:\Users\hari1996\Nextcloud\Shared\14_DGGT\01_Figures\01_Fig_su_G0\Plotting.xlsx"

SU_COLS = [
    "su_CU_undisturbed (kPa)", "su_CUe_undisturbed (kPa)", "su_DSS (kPa)",
    "su_UU_remoulded (kPa)", "su_UU_undisturbed (kPa)", "su_FCT (kPa)",
]

# su reference series: (value column, depth column)
SU_REF_SERIES = [
    ("CIUC",    "z_CIUC"),
    ("CAUC",    "z_CAUC"),
    ("Vane83",  "z_Vane83"),
    ("VaneGSC", "z_VaneGSC"),
    ("Vane72",  "z_Vane72"),
]

# G0 reference series (MPa), converted to Vs: (value column, depth column)
G0_REF_SERIES = [
    ("Vs_02",     "z_02"),
    ("Vs_14",     "z_14"),
    ("Vs_Dejong", "z_Dejong"),
    ("Vs_Land",   "z_Land"),
    ("Vs_Mayne",  "z_Mayne"),
]

# single, consistent marker style for all reference points (kept distinct
# from the density colormap so it stays legible on top of it)
REF_MARKER = "."
REF_FACE = "none"
REF_EDGE = "#d62728"
REF_LABEL = "Reference values"

VS_XLIM = (0, 1000)
SU_XLIM = (0, 100)
DEPTH_XLIM = (0, 22)
VS2_XLIM = (0,500)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ---------------------------------------------------------------------------
# Vs (m/s) -- Dataset 1, same physical filter as the other Dataset 1 figures
# ---------------------------------------------------------------------------
df1 = pd.read_parquet(INPUT_DATASET1, columns=["Depth (m)", "Vs (m/s)"])
df1["Vs (m/s)"] = pd.to_numeric(df1["Vs (m/s)"], errors="coerce")
df1["Depth (m)"] = pd.to_numeric(df1["Depth (m)"], errors="coerce")
df1 = df1[(df1["Vs (m/s)"] > 0) & (df1["Vs (m/s)"] < 1500)].dropna()
vs, vs_depth = df1["Vs (m/s)"], df1["Depth (m)"]

# ---------------------------------------------------------------------------
# Undrained shear strength su (kPa) -- Dataset 2, pooled across lab test types
# ---------------------------------------------------------------------------
df2 = pd.read_parquet(INPUT_DATASET2, columns=["Depth (m)"] + SU_COLS)
df2["Depth (m)"] = pd.to_numeric(df2["Depth (m)"], errors="coerce")
su_parts, depth_parts = [], []
for c in SU_COLS:
    vals = pd.to_numeric(df2[c], errors="coerce")
    sub = pd.DataFrame({"su": vals, "depth": df2["Depth (m)"]})
    sub = sub[sub["su"] > 0].dropna()
    su_parts.append(sub["su"])
    depth_parts.append(sub["depth"])
su = pd.concat(su_parts, ignore_index=True)
su_depth = pd.concat(depth_parts, ignore_index=True)

# ---------------------------------------------------------------------------
# Literature reference points (DGGT su/G0 case-study figure)
# ---------------------------------------------------------------------------
df_ref = pd.read_excel(REF_EXCEL, sheet_name="Ref")

su_ref_vals, su_ref_depths = [], []
for val_col, depth_col in SU_REF_SERIES:
    sub = df_ref[[val_col, depth_col]].dropna()
    su_ref_vals.append(sub[val_col].values)
    su_ref_depths.append(sub[depth_col].values)
su_ref_vals = np.concatenate(su_ref_vals)
su_ref_depths = np.concatenate(su_ref_depths)

# site's own measured unit weight profile (tube samples) -> rho(depth), for
# converting the G0 reference points to Vs via Vs = sqrt(G0 / rho)
unit_profile = df_ref[["z_unit", "Unit"]].dropna().sort_values("z_unit")

vs_ref_vals, vs_ref_depths = [], []
for val_col, depth_col in G0_REF_SERIES:
    sub = df_ref[[val_col, depth_col]].dropna()
    if len(sub) == 0:
        continue
    gamma = np.interp(sub[depth_col].values, unit_profile["z_unit"].values, unit_profile["Unit"].values)  # kN/m3
    rho = gamma * 1000 / 9.81  # kg/m3
    vs_ref_vals.append(np.sqrt(sub[val_col].values * 1e6 / rho))  # m/s
    vs_ref_depths.append(sub[depth_col].values)
vs_ref_vals = np.concatenate(vs_ref_vals)
vs_ref_depths = np.concatenate(vs_ref_depths)


def smooth_density(ax, x, y, xlim, ylim, vmax=None, n_bins=60, sigma=1.5, cmap="Greys"):
    """Fine 2D histogram of (x, y) smoothed with a Gaussian filter and
    rendered as a continuous density surface (pcolormesh, gouraud shading).
    vmax caps the color scale, so a few dense cells don't wash out the rest."""
    x_edges = np.linspace(*xlim, n_bins + 1)
    y_edges = np.linspace(*ylim, n_bins + 1)
    counts, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges])
    counts = gaussian_filter(counts, sigma=sigma)
    x_centers = (xe[:-1] + xe[1:]) / 2
    y_centers = (ye[:-1] + ye[1:]) / 2
    return ax.pcolormesh(x_centers, y_centers, counts.T, cmap=cmap, shading="gouraud",
                          vmin=0, vmax=vmax)


# ---------------------------------------------------------------------------
# Figure: 3x2 grid -- value histograms, depth histograms, depth profiles
# ---------------------------------------------------------------------------
def build_figure(overlay_ref, out_basename):
    cm = 1 / 2.54
    fig, axes = plt.subplots(3, 2, figsize=(16 * cm, 20 * cm), dpi=200, )

    axes[0, 0].hist(su, bins=np.linspace(*SU_XLIM, 31), color="0.4", edgecolor="white", linewidth=0.3)
    axes[0, 0].set_xlabel(r"$s_u$ (kPa)")
    axes[0, 0].set_ylabel("Counts")
    axes[0, 0].set_xlim(SU_XLIM)
    axes[0, 0].text(0.97, 0.95, f"N={len(su):,}", transform=axes[0, 0].transAxes,
                    ha="right", va="top", fontsize=8)

    axes[0, 1].hist(vs, bins=np.linspace(*VS_XLIM, 31), color="0.4", edgecolor="white", linewidth=0.3)
    axes[0, 1].set_xlabel(r"$V_s$ (m/s)")
    axes[0, 1].set_ylabel("Counts")
    axes[0, 1].set_xlim(VS_XLIM)
    axes[0, 1].text(0.97, 0.95, f"N={len(vs):,}", transform=axes[0, 1].transAxes,
                    ha="right", va="top", fontsize=8)

    axes[1, 0].hist(su_depth, bins=np.linspace(*DEPTH_XLIM, 31), color="0.4", edgecolor="white", linewidth=0.3)
    axes[1, 0].set_xlabel("Depth (m)")
    axes[1, 0].set_ylabel("Counts")
    axes[1, 0].set_xlim(DEPTH_XLIM)

    axes[1, 1].hist(vs_depth, bins=np.linspace(*DEPTH_XLIM, 31), color="0.4", edgecolor="white", linewidth=0.3)
    axes[1, 1].set_xlabel("Depth (m)")
    axes[1, 1].set_ylabel("Counts")
    axes[1, 1].set_xlim(DEPTH_XLIM)

    im0 = smooth_density(axes[2, 0], su, su_depth, SU_XLIM, DEPTH_XLIM, vmax=1.0)
    axes[2, 0].set_xlabel(r"$s_u$ (kPa)")
    axes[2, 0].set_ylabel("Depth (m)")
    axes[2, 0].set_xlim(SU_XLIM)
    axes[2, 0].set_ylim(DEPTH_XLIM)
    axes[2, 0].invert_yaxis()
    fig.colorbar(im0, ax=axes[2, 0], label="Density", fraction=0.046, pad=0.04)

    im1 = smooth_density(axes[2, 1], vs, vs_depth, VS_XLIM, DEPTH_XLIM, vmax=60)
    axes[2, 1].set_xlabel(r"$V_s$ (m/s)")
    axes[2, 1].set_ylabel("Depth (m)")
    axes[2, 1].set_xlim(VS2_XLIM)
    axes[2, 1].set_ylim(DEPTH_XLIM)
    axes[2, 1].invert_yaxis()
    fig.colorbar(im1, ax=axes[2, 1], label="Density", fraction=0.046, pad=0.04)

    if overlay_ref:
        for ax, values, depths in [(axes[2, 0], su_ref_vals, su_ref_depths),
                                    (axes[2, 1], vs_ref_vals, vs_ref_depths)]:
            ax.scatter(values, depths, marker=REF_MARKER, fc=REF_FACE, ec=REF_EDGE,
                       s=8, lw=0.8, zorder=5, label=REF_LABEL)
            ax.legend(fontsize=11, loc="lower right", framealpha=0.85, bbox_to_anchor=(1, -0.45), frameon = True)

    plt.subplots_adjust(left=0.10, right=0.90, bottom = 0.10, top = 0.99, wspace= 0.6, hspace=0.3)
    fig.savefig(f"{out_basename}.png", dpi=1000)
    plt.close(fig)
    print(f"Saved {out_basename}.png (Vs N={len(vs):,}, su N={len(su):,})")


build_figure(overlay_ref=False, out_basename=OUT_BASENAME)
build_figure(overlay_ref=True, out_basename=f"{OUT_BASENAME}_with_ref")

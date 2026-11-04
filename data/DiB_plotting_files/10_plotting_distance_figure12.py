"""
dataset2_horizontal_distance.py

Figure for the Data Description section: horizontal distance between the
CPT sounding and the paired lab sample, compared across sites. This is a
data-quality indicator for the "coupled" part of Dataset 2 — smaller
distance means a tighter, more reliable CPT/lab pairing at that site.

Design choice: ONE shared axis, not small multiples. The small-multiples
raincloud-per-site version (each site its own independent axis) matched
the style of the other Dataset 2 figures but made cross-site comparison
hard — you couldn't tell at a glance that Australia's offsets are an order
of magnitude larger than the Dutch sites', because every panel rescaled
independently. A ranked strip comparison on one shared (symlog) axis is
the fix: sites ordered by median distance so the pattern is visible
immediately, without needing a box/whisker summary on top of it.

Each site: individual points jittered vertically, single color throughout
— every data point stays visible, nothing is summarized into a median,
box, or any other statistic. A light horizontal guide line runs across
each row (including the two empty ones) so a point cluster can be traced
back to its row label even when rows sit close together — otherwise, at
14 rows in a compact figure, it's easy to lose track of which dots belong
to which site. Symlog x-axis (linear near zero, log beyond) handles the
mix of exact-zero and up-to-158 m values.

Only 554 of 1485 records carry this metadata. Two sites (NGTS Norway and
Offshore Wind RVO NL) have none at all and are listed at the bottom with
"n=0" rather than silently omitted, so the gap stays visible.

Output: dataset2_horizontal_distance.png, sized to Elsevier's double-column
width (19 cm).

Update INPUT_XLSX below if the file moves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

cm = 1 / 2.54
rng = np.random.default_rng(42)

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
INPUT_PARQUET = "..\\11_databases\\Dataset2_Coupled_Geotechnical_CPT_20260730_withSBT.parquet"
OUT_BASENAME = "Figure12_dataset2_horizontal_distance"
# ----------------------------------------------------------------------------

df = pd.read_parquet(INPUT_PARQUET, columns=["sampling_name (-)", "Horizontal distance (m)"])
df = df.replace(r"^\s*$", np.nan, regex=True)


def site_group(name):
    if pd.isna(name):
        return None
    n = str(name).strip()
    mapping = [
        ("Austria_site", "Austria (private)"),
        ("Ballina", "Australia"),
        ("Bergambacht", "Bergambacht, NL"),
        ("DOV", "DOV, NL"),
        ("Delft", "Delft, NL"),
        ("Full Scale Damwand Proef Eemdijk", "D.P. Eemdijk, NL"),
        ("HKW", "Offshore Wind RVO, NL"),
        ("Kinderdijk", "Kinderdijk, NL"),
        ("Koehool", "Koehool, NL"),
        ("MMD Katwoude", "MMD Katwoude, NL"),
        ("MMD Pieterman", "MMD Pieterman, NL"),
        ("NGTS", "NGTS, Norway"),
        ("NWZ", "Offshore Wind RVO, NL"),
        ("Norway_site", "Norway (private)"),
        ("PITS_Rhesi", "Austria (private)"),
        ("PITS_", "Austria (private)"),
        ("PTS_", "Austria (private)"),
        ("Salzburg", "Austria (private)"),
        ("Schalkwijk", "Schalkwijk, NL"),
        ("Streefkerk", "Streefkerk, NL"),
        ("TNW", "Offshore Wind RVO, NL"),
        ("Waardenburg", "Waardenburg, NL"),
    ]
    for prefix, group in mapping:
        if n.startswith(prefix):
            return group
    return "Other/unmapped"


df["site"] = df["sampling_name (-)"].apply(site_group)

all_sites = [
    "Austria (private)", "Australia",
    "Bergambacht, NL", "DOV, NL", "Delft, NL", "D.P. Eemdijk, NL",
    "Kinderdijk, NL", "Koehool, NL", "MMD Katwoude, NL", "MMD Pieterman, NL",
    "Schalkwijk, NL", "Streefkerk, NL", "Waardenburg, NL",
    "Offshore Wind RVO, NL", "NGTS, Norway", "Norway (private)",
]
all_sites = [s for s in all_sites if s in df["site"].unique()]

# --- pull per-site data, rank sites with data by median (ascending) --------
site_data = {}
for s in all_sites:
    d = pd.to_numeric(df.loc[df["site"] == s, "Horizontal distance (m)"], errors="coerce").dropna()
    site_data[s] = d.values

sites_with_data = [s for s in all_sites if len(site_data[s]) > 0]
sites_with_data.sort(key=lambda s: np.median(site_data[s]))
sites_empty = [s for s in all_sites if len(site_data[s]) == 0]
site_order = sites_with_data + sites_empty  # empty ones fall to the bottom

fig_w = 19 * cm
fig_h = 6 * cm
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=500)

y_positions = {s: i for i, s in enumerate(site_order)}

POINT_COLOR = cmc.batlowS.colors[6]

# horizontal guide line per row (including empty ones), drawn first/behind
# everything else, so each row's points/label stay traceable at a glance
for s in site_order:
    ax.axhline(y_positions[s], color="#d9d9d9", linewidth=0.6, zorder=1)

# jittered points only, single color
for s in sites_with_data:
    d = site_data[s]
    y0 = y_positions[s]
    jitter = rng.uniform(-0.28, 0.28, size=len(d))
    ax.scatter(d, y0 + jitter, s=10, alpha=0.5, color=POINT_COLOR, linewidths=0, zorder=3)

    ax.annotate(f"n={len(d)}", xy=(1.0, y0), xycoords=("axes fraction", "data"),
                 xytext=(4, 0), textcoords="offset points",
                 va="center", fontsize=6, color="#555555")

for s in sites_empty:
    y0 = y_positions[s]
    ax.text(0, y0, "  n=0 (no distance recorded)", va="center", ha="left",
             fontsize=6.5, color="#888888", style="italic")

ax.set_yticks(range(len(site_order)))
ax.set_yticklabels(site_order)
ax.set_ylim(-0.7, len(site_order) - 0.3)
ax.invert_yaxis()
ax.set_xscale("symlog", linthresh=1)
ax.set_xlim(left=-0.3)
ax.set_xlabel("Distance (m)")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.subplots_adjust(
    left=0.20,
    right=0.95,
    top=0.99,
    bottom=0.19,
    wspace=0.4,
    hspace=0.5
)

fig.savefig(f"{OUT_BASENAME}.png", dpi=500)
print(f"Saved {OUT_BASENAME}.png")
print("Sites ranked by median horizontal distance:")
for s in sites_with_data:
    print(f"  {s:25s} median={np.median(site_data[s]):.2f} m  n={len(site_data[s])}")
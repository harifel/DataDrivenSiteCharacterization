import json
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import shape

# ---------------------------------------------------------------------------
# 1. Per-site totals computed directly from Dataset2_..._.xlsx, replacing the
#    hand-transcribed Table-4 approximation. This also resolves the two
#    CHECK items from the previous version:
#      - Rhesi is an Austrian PITS_ sampling series (sampling_name prefix
#        "PITS_Rhesi") -> folded into "Austria (private)", not a separate
#        unconfirmed country.
#      - Offshore Wind RVO is three sub-projects (HKW, NWZ, TNW) grouped
#        under one Netherlands entry, matching Dataset 1's single listing.
#    "n_records" here is the same quantity as the heatmap's Total column:
#    the sum of non-missing values across all coupled lab/index parameters
#    for that site (not raw row count) — kept consistent with how the
#    original script summed several per-parameter figures per site.
# ---------------------------------------------------------------------------
# Updated 2026-07-28: loads the Parquet conversion of the withSBT workbook
# instead of the xlsx directly -- see 15_python scripts/convert_dataset2_to_parquet.py.
# INPUT_PARQUET = "..\\11_databases\\Dataset2_Coupled_Geotechnical_CPT_20260728.parquet"
INPUT_PARQUET = "..\\11_databases\\Dataset2_Coupled_Geotechnical_CPT_20260730_withSBT.parquet"

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
df = df.replace(r"^\s*$", np.nan, regex=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize":  9,
    "axes.grid":       False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 9,
})



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

param_cols = [c for c in param_cols_all if c in df.columns]

site_order = [
    "Austria (private)", "Australia",
    "Bergambacht, NL", "DOV, NL", "Delft, NL", "D.P. Eemdijk, NL",
    "Kinderdijk, NL", "Koehool, NL", "MMD Katwoude, NL", "MMD Pieterman, NL",
    "Schalkwijk, NL", "Streefkerk, NL", "Waardenburg, NL",
    "Offshore Wind RVO, NL", "NGTS, Norway", "Norway (private)",
]
site_order = [s for s in site_order if s in df["site"].unique()]

counts = (
    df[df["site"].isin(site_order)]
    .groupby("site")[param_cols]
    .apply(lambda g: g.notna().sum())
    .reindex(site_order)
)
# drop parameter columns with zero records across every site (e.g. su_CU_remoulded)
counts = counts.drop(columns=["E50ref_m_parameter (-)"], errors="ignore")  # <-- add this line
counts = counts.loc[:, counts.sum(axis=0) > 0]

site_totals = counts.sum(axis=1).to_dict()

site_to_country = {
    "Austria (private)": "Austria",
    "Australia": "Australia",
    "Bergambacht, NL": "Netherlands",
    "DOV, NL": "Netherlands",
    "Delft, NL": "Netherlands",
    "D.P. Eemdijk, NL": "Netherlands",
    "Kinderdijk, NL": "Netherlands",
    "Koehool, NL": "Netherlands",
    "MMD Katwoude, NL": "Netherlands",
    "MMD Pieterman, NL": "Netherlands",
    "Schalkwijk, NL": "Netherlands",
    "Streefkerk, NL": "Netherlands",
    "Waardenburg, NL": "Netherlands",
    "Offshore Wind RVO, NL": "Netherlands",
    "NGTS, Norway": "Norway",
    "Norway (private)": "Norway",
}

df_sites = pd.DataFrame({
    "site": list(site_totals.keys()),
    "n_records": list(site_totals.values()),
})
df_sites["country"] = df_sites["site"].map(site_to_country)

country_totals = df_sites.groupby("country")["n_records"].sum().reset_index()
print(country_totals)
print("Grand total:", country_totals["n_records"].sum())

# ---------------------------------------------------------------------------
# 2. Load world boundaries and merge
# ---------------------------------------------------------------------------
# NOTE: Using json + shapely instead of gpd.read_file() here, because some
# restricted/bundled Python distributions (e.g. PLAXIS internal Python) ship
# a fiona version incompatible with their geopandas version, causing:
#   AttributeError: module 'fiona' has no attribute 'path'
# This loader avoids fiona entirely. Adjust the path below to wherever your
# world.geojson file is stored.
geojson_path = "world.geojson"  # e.g. r"C:\...\12_figures\world.geojson"

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

records = []
for feature in geojson_data["features"]:
    props = dict(feature["properties"])
    props["geometry"] = shape(feature["geometry"])
    records.append(props)

world = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
world = world.merge(country_totals, left_on="ADMIN", right_on="country", how="left")

# Diagnostic: flag any country_totals entries that did NOT find a match in
# the geojson's ADMIN field (e.g. name spelling differs, or the country is
# missing from this particular world.geojson export).
matched = set(world.loc[world["n_records"].notna(), "country"].dropna())
unmatched = set(country_totals["country"]) - matched
if unmatched:
    print(f"WARNING: no ADMIN match found in world.geojson for: {sorted(unmatched)}")
    print("Check the exact spelling used in your geojson, e.g.:")
    admin_names = world["ADMIN"].dropna().unique()
    for c in unmatched:
        close = [a for a in admin_names if c.lower()[:4] in a.lower()]
        print(f"  '{c}' -> candidates in file: {close}")

# ---------------------------------------------------------------------------
# 3. Plot: flat highlight color for countries with data, number labeled at
#    centroid, NO colorbar/legend
# ---------------------------------------------------------------------------
cm = 1 / 2.54  # inches per cm, so sizes below can be specified in cm

fig_width_in = 19.0 * cm    # double-column width per Elsevier guidelines (190 mm)
fig_height_in = 8 * cm
dpi = 1000

fig, ax = plt.subplots(1, 1, figsize=(fig_width_in, fig_height_in), dpi=dpi)

world.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.3)

has_data = world[world["n_records"].notna()]
has_data.plot(ax=ax, color="#92c5de", edgecolor="#333333", linewidth=0.5)

# Manual label offsets (in map degrees) for countries whose default centroid
# position is too small/crowded to read clearly (Netherlands and Austria sit
# close together in this dataset too).
label_offsets = {
    "Netherlands": (-10, 6, True),
    "Austria": (8, -4, True),
}

for _, row in has_data.iterrows():
    geom = row["geometry"]
    # For MultiPolygons (e.g. Norway = mainland + Svalbard + Jan Mayen),
    # representative_point() on the whole geometry can land on a remote
    # island far outside the plotted extent (Svalbard is ~80N, above this
    # map's ylim=75), which silently clips the label. Use the largest
    # sub-polygon (the mainland) instead.
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    point = geom.representative_point()
    anchor_x, anchor_y = point.x, point.y

    dx, dy, use_leader = label_offsets.get(row["country"], (0, 0, False))
    label_x, label_y = anchor_x + dx, anchor_y + dy

    if use_leader:
        ax.plot(
            [anchor_x, label_x], [anchor_y, label_y],
            color="#333333", linewidth=0.6, zorder=4,
        )
        ax.plot(anchor_x, anchor_y, marker="o", markersize=2.5, color="#333333", zorder=5)

    txt = ax.annotate(
        text=f"{int(row['n_records']):,}",
        xy=(label_x, label_y),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#1a1a1a",
        zorder=6,
    )
    txt.set_path_effects([
        path_effects.Stroke(linewidth=2.5, foreground="white"),
        path_effects.Normal(),
    ])

ax.set_title("Dataset 2: Geographic distribution of coupled CPT + laboratory records", fontsize=9, fontweight="bold")
ax.set_axis_off()
ax.set_xlim(-180, 180)
ax.set_ylim(-50, 75)


fig.subplots_adjust(
    left=0.00,
    right=1,
    top=0.94,
    bottom=0.01,
    wspace=0.05,
    hspace=0.03
)


plt.savefig("Figure6_dataset2_world_map.png", dpi=dpi, format="png")

"""
Version 2 of the Dataset 1 world map figure, with the following changes from v1:"
    uses Dataset1_CPT_database_FINAL_v3.csv instead of v2"
    uses country column instead of assigning manually
       --> (source code: 11_databases\archive\_03_country_assignment.py)"

Updated 2026-07-28 to load Dataset1_CPT_database_20260728.parquet instead of
the v3 CSV: this is the merged database (Dataset 1 + Dataset 3/Geoprofile,
Switzerland — see 15_python scripts/merge_dataset1_dataset3.py), with SBT
recomputed and the "Netherlands (1)" -> "Netherlands (Vardon)" label fix
already applied. Parquet also loads far faster than the 1.8 GB CSV since it's
columnar/compressed and only the 4 needed columns get read.
"""
#%%
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
from shapely.geometry import shape

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize":  9,
    "axes.grid":       False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 9,
})
#%%

# ---------------------------------------------------------------------------
# 1. Per-record counts computed directly from the VERIFIED v5 compiled CSV
#    (Nepal test types corrected, Rhesi merged in, Snakes/RVO duplication
#    resolved, NZGD's SCPT7xxx group correctly attributed - see the full
#    audit trail for details).
#
# 2026-07-28: removed the old database-name -> country lookup (row_to_database
# / database_totals / unmapped_count / database_order) that used to live here.
# It was dead code -- nothing downstream read it, and the docstring already
# says this version's whole point is to read country directly from the
# "country" column instead of assigning it manually. Only "ID" and "country"
# are needed now.
# ---------------------------------------------------------------------------
INPUT = "..\\11_databases\\Dataset1_CPT_database_20260728.parquet"

TEST_MODE = False
dpi = 150 if TEST_MODE else 1000

df = pd.read_parquet(INPUT, columns=["ID", "country"])
df = df.replace(r"^\s*$", np.nan, regex=True)

#%%

# ---------------------------------------------------------------------------
# Aggregate soundings by country directly from the verified country column.
# Countries without an assigned mapping are grouped as "Unknown / Private".
# ---------------------------------------------------------------------------
country_totals = (
    df.assign(country=df["country"].fillna("Unknown / Private"))
      .groupby("country")["ID"]
      .nunique()
      .reset_index(name="n_soundings")
)

print(country_totals)
print("Grand total:", country_totals["n_soundings"].sum())

# Geojson country names are not always identical to the country names in the CSV.
PLOT_COUNTRY_NAMES = {"Türkiye": "Turkey",
}

country_totals["country"] = (country_totals["country"].replace(PLOT_COUNTRY_NAMES))


# ---------------------------------------------------------------------------
# 2. Load world boundaries and merge
# ---------------------------------------------------------------------------
geojson_path = "world.geojson"

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

records = []
for feature in geojson_data["features"]:
    props = dict(feature["properties"])
    props["geometry"] = shape(feature["geometry"])
    records.append(props)

world = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
world = world.merge(country_totals, left_on="ADMIN", right_on="country", how="left")

matched = set(world.loc[world["n_soundings"].notna(), "country"].dropna())
unmatched = set(country_totals["country"]) - matched
if unmatched:
    print(f"WARNING: no ADMIN match found in world.geojson for: {sorted(unmatched)}")
    admin_names = world["ADMIN"].dropna().unique()
    for c in unmatched:
        close = [a for a in admin_names if c.lower()[:4] in a.lower()]
        print(f"  '{c}' -> candidates in file: {close}")

#%%
# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
cm = 1 / 2.54

fig_width_in = 19.0 * cm
fig_height_in = 8 * cm

fig, ax = plt.subplots(1, 1, figsize=(fig_width_in, fig_height_in), dpi=dpi)

world.plot(ax=ax, color="#f0f0f0", edgecolor="#999999", linewidth=0.3)

has_data = world[world["n_soundings"].notna()]
has_data.plot(ax=ax, color="#f4a582", edgecolor="#333333", linewidth=0.5)

label_offsets = {
    "Netherlands": (-10, 6, True),
    "Austria": (7, 7, True),
    "Italy": (2, -10, True),
    "Nepal": (0, -10, True),
    "Taiwan": (12, -6, True),
    "Norway": (10, 4, True),
    "Turkey": (10, -10, True),
    "Mexico": (-10, -10, True),
    "Switzerland": (-10, -10, True),
}

# Any country not manually configured above but with a "minor" count still
# gets an automatic dot + leader line, so small highlighted areas never end
# up with a number sitting directly (and illegibly) on top of them.
MINOR_THRESHOLD = 100
DEFAULT_MINOR_OFFSET = (8, 8, True)  # (dx, dy, use_leader) in map degrees

for _, row in has_data.iterrows():
    country = row["country"]
    if country not in label_offsets and row["n_soundings"] < MINOR_THRESHOLD:
        label_offsets[country] = DEFAULT_MINOR_OFFSET

for _, row in has_data.iterrows():
    geom = row["geometry"]
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
        text=f"{int(row['n_soundings']):,}",
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

ax.set_title("Dataset 1: Geographic distribution of CPT soundings", fontsize=9, fontweight="bold")
ax.set_axis_off()
ax.set_xlim(-180, 180)
ax.set_ylim(-50, 75)

fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.05, hspace=0.03)

out_name = "dataset1_world_map_TEST_02.png" if TEST_MODE else "Figure1_dataset1_world_map_02.png"
plt.savefig(out_name, dpi=dpi, format="png")
# %%



#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
})

# -----------------------------
# Constants
# -----------------------------

PA_MPA = 0.10135          # atmospheric pressure [MPa]
PA_KPA = PA_MPA * 1000     # atmospheric pressure [kPa]

CM = 1 / 2.54
FIG_WIDTH_CM = 11
FIG_HEIGHT_CM = 11

CHART_XLIM = (0.1, 10)      # Rf (%)
CHART_YLIM = (1, 1000)      # qt/pa (-)

# Cluster labels and their positions on the Robertson (1986, 2010) chart
CLUSTER_LABELS = [
    (0.3, 2.5, "1"),
    (7,   1.5, "2"),
    (4.5, 7.0, "3"),
    (2.6, 20,  "4"),
    (1.8, 50,  "5"),
    (1.3, 200, "6"),
    (0.25, 500, "7"),
    (3.0, 500, "8"),
    (7,   160, "9"),
]

IMAGE_PATH = r"i_Robertson 2010_lines.png"
SAVE_PATH = r"Figure13_Robertson_plot.png"


# -----------------------------
# Helper functions
# -----------------------------

def _draw_robertson_chart(ax, image_path):
    """
    Draw the static parts of the Robertson (1986, updated 2010) chart:
    background image, axis ticks/limits, grid, and cluster labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the chart on.
    image_path : str
        Path to the Robertson diagram background image.
    """
    lxmin, lxmax = np.log10(CHART_XLIM)
    lymin, lymax = np.log10(CHART_YLIM)

    im = image.imread(image_path)
    ax.imshow(im, extent=(lxmin, lxmax, lymin, lymax), zorder=0, aspect="auto")

    # Major ticks
    ax.set_xticks(np.log10([0.1, 1, 10]))
    ax.set_xticklabels([0.1, 1, 10])
    ax.set_yticks(np.log10([1, 10, 100, 1000]))
    ax.set_yticklabels([1, 10, 100, 1000])
    ax.set_xlim(lxmin, lxmax)
    ax.set_ylim(lymin, lymax)

    # Minor ticks (log-scale style, without using a true log axis)
    minor_xticks = np.log10(np.concatenate([np.arange(2, 10) / 10, np.arange(2, 10)]))
    minor_yticks = np.log10(np.concatenate([np.arange(2, 10) * base for base in (1, 10, 100)]))
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    for x, y, label in CLUSTER_LABELS:
        ax.text(np.log10(x), np.log10(y), label, fontsize=8, color="k",
                 va="center", ha="center")

    ax.grid(which="major", lw=0.1, c="grey", ls="dotted")
    ax.grid(which="minor", lw=0.1, c="grey", ls="dotted")
    ax.set_xlabel("$R_f~(\\%)$", labelpad=3)
    ax.set_ylabel("$q_{t} / p_a~(-)$", labelpad=-5)


def add_metadata_footnote(fig, df, id_col="ID", db_col="database"):
    """
    Add a footnote below the plot listing the CPT sounding ID, its data
    source (database), and the atmospheric pressure pa used to normalize qt.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to annotate.
    df : pandas.DataFrame
        Data being plotted. Missing `id_col`/`db_col` are handled gracefully.
    id_col, db_col : str
        Column names for the sounding ID and data source.

    Notes
    -----
    If `df` contains multiple unique IDs or databases (e.g. several
    soundings plotted together), all unique values are listed.
    """
    id_str = ", ".join(map(str, df[id_col].unique())) if id_col in df.columns else "n/a"
    db_str = ", ".join(map(str, df[db_col].unique())) if db_col in df.columns else "n/a"

    footnote = (
        f"ID: {id_str}   |   Database: {db_str}\n"
        f"$p_{{a}}$: atmospheric pressure (101.3 kPa)"
    )
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=9)


def get_random_ID(df, id_col="ID", type_col="test type", valid_types=("CPTu", "SCPTu")):
    """
    Pick a random CPT ID from `df`, restricted to `valid_types`.

    Parameters
    ----------
    df : pandas.DataFrame
    id_col : str, default "ID"
    type_col : str, default "test type"
    valid_types : tuple of str, default ("CPTu", "SCPTu")

    Returns
    -------
    random_id : str or int
    df_subset : pandas.DataFrame
        Rows of `df` matching `random_id`.
    """
    valid_ids = df.loc[df[type_col].isin(valid_types), id_col].unique()
    if len(valid_ids) == 0:
        raise ValueError("No IDs found with the given test types.")
    random_id = np.random.choice(valid_ids)
    return random_id, df[df[id_col] == random_id]


# -----------------------------
# Main plotting function
# -----------------------------

def plot_robertson_with_depth(image_path, df, depth_column=None, save_path=None,
                               id_col="ID", db_col="database"):
    """
    Plot CPT data (qt, Rf) on the Robertson (1986, updated 2010) chart,
    with points coloured by depth.

    Parameters
    ----------
    image_path : str
        Path to the Robertson diagram background image.
    df : pandas.DataFrame
        Must contain 'qc (MPa)', 'Rf (%)', and a depth column.
    depth_column : str, optional
        Column to use for depth. Defaults to the first column containing "depth".
    save_path : str, optional
        If given, saves the figure to this path as a PNG.
    id_col, db_col : str
        Column names used for the ID/database footnote (see `add_metadata_footnote`).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Notes
    -----
    - Points are coloured by depth using the 'viridis' colormap; the
      colorbar is inverted so depth increases downward.
    - qt/pa is plotted on a log scale (via manually placed log10 ticks),
      normalized by the atmospheric pressure pa.
    - A footnote below the plot shows the sounding ID, data source, and pa.
    """
    if depth_column is None:
        depth_candidates = [col for col in df.columns if "depth" in col.lower()]
        if not depth_candidates:
            raise ValueError("No depth column found. Please specify `depth_column`.")
        depth_column = depth_candidates[0]

    qt_pa_log = np.log10(df["qc (MPa)"] * 1000 / PA_KPA)
    Rf_log = np.log10(df["Rf (%)"])
    depth = df[depth_column].values

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_CM * CM, FIG_HEIGHT_CM * CM), dpi=200)
    _draw_robertson_chart(ax, image_path)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.18)

    sc = ax.scatter(
        Rf_log, qt_pa_log,
        c=depth, cmap="viridis",
        s=5, marker="o", edgecolors="none", alpha=0.8, zorder=2,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(depth_column, labelpad=5)
    cbar.ax.invert_yaxis()

    add_metadata_footnote(fig, df, id_col=id_col, db_col=db_col)

    if save_path is not None:
        fig.savefig(save_path, dpi=200)

    return fig, ax


#%% Read dataset

DF_DATASET1 = pd.read_parquet("..\\11_databases\\Dataset1_CPT_database_20260728.parquet")

#%% Random sounding

random_id, df_random_choice = get_random_ID(DF_DATASET1, valid_types=("CPTu", "SCPTu"))
print("Random ID:", random_id)
plot_robertson_with_depth(IMAGE_PATH, df_random_choice, save_path=None)

#%% Specific sounding

spec_id = "PRJ_2788_CPT-DAL-45TT"  # e.g. "TNW057-PCPT", "PRJ_2788_CPT-DAL-45TT"
df_spec_choice = DF_DATASET1[DF_DATASET1["ID"] == spec_id]

plot_robertson_with_depth(IMAGE_PATH, df_spec_choice, save_path=SAVE_PATH)

# %%
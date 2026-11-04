#%% =============================================================================
# Robertson (1986, updated 2010) chart: density plots for Dataset 1 and 2
# =============================================================================
#
# For each dataset this script can produce two figures:
#
#   1. Density contour: a kernel density estimate (KDE) of a random sample of
#      CPT points, drawn as filled contours. Shows where the data concentrate
#      on the chart. Colour is probability density per unit area in log space.
#
#   2. 2D density heatmap: a histogram of ALL points, binned on a 40 x 40 grid.
#      Colour is the number of points per bin, so it shows absolute counts.
#
# Both plots use the Robertson chart axes:
#   x = Rf (%)       on a log10 scale
#   y = qt / pa (-)  on a log10 scale
#
# qt = qc + u2 * (1 - a), with net area ratio a = 0.80. pa is atmospheric
# pressure.


#%% -----------------------------------------------------------------------------
# 0. Imports
# -----------------------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import cmcrameri.cm as cmc

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
})


#%% -----------------------------------------------------------------------------
# 1. Settings
# -----------------------------------------------------------------------------

DATA_DIR = os.path.join("..", "11_databases")
DATASET1_PATH = os.path.join(DATA_DIR, "Dataset1_CPT_database.parquet")
DATASET2_PATH = os.path.join(DATA_DIR, "Dataset2_Coupled_Geotechnical_CPT.parquet")
IMAGE_PATH = "i_Robertson 2010_lines.png"

# Folder for saved figures: the folder of this script, or the current
# working directory when the script is run without a file (e.g. pasted
# into a console).
try:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SAVE_DIR = os.getcwd()

SAVE_DPI = 300

N_SAMPLES = 16000        # points used for the density contour (KDE cost grows with N)
RANDOM_STATE = 42        # fixed seed so the sample is reproducible

PA_KPA = 0.10135 * 1000  # atmospheric pressure [kPa]
NET_AREA_RATIO = 0.80    # a in qt = qc + u2 * (1 - a)

CM = 1 / 2.54
FIG_SIZE_CM = (11, 11)

CHART_XLIM = (0.1, 10)   # Rf (%)
CHART_YLIM = (1, 1000)   # qt / pa (-)
LOG_XLIM = np.log10(CHART_XLIM)
LOG_YLIM = np.log10(CHART_YLIM)

# Zone numbers and their positions on the Robertson (1986, 2010) chart
CLUSTER_LABELS = [
    (0.3, 2.5, "1"),
    (7, 1.5, "2"),
    (4.5, 7.0, "3"),
    (2.6, 20, "4"),
    (1.8, 50, "5"),
    (1.1, 200, "6"),
    (0.25, 500, "7"),
    (3.0, 500, "8"),
    (7, 160, "9"),
]

# Hot colormap that ends in yellow instead of white, so the densest regions
# stay visible against the light chart background.
HOT_NOWHITE = LinearSegmentedColormap.from_list(
    "hot_nowhite",
    ["#000000", "#8b0000", "#e0201a", "#ff8c00", "#f8f52b"],
)


# Perceptually uniform colormap from cmcrameri, consistent with the other
# figures. It ends in yellow, so the densest regions stay visible against
# the light chart background.
DENSITY_CMAP = cmc.batlow

#%% -----------------------------------------------------------------------------
# 2. Data helpers
# -----------------------------------------------------------------------------

def load_cpt_points(path):
    """
    Load a CPT database and keep rows usable for the Robertson chart.

    Rows are kept only if qc, Rf and u2 are present and qc > 0 and Rf > 0
    (both are needed for the log transforms).

    Parameters
    ----------
    path : str
        Path to a parquet file with 'qc (MPa)', 'Rf (%)' and 'u2 (MPa)'.

    Returns
    -------
    pandas.DataFrame
        Filtered data with a fresh index.
    """
    # df = pd.read_parquet(path)
    # df = df.dropna(subset=["qc (MPa)", "Rf (%)", "u2 (MPa)"])
    
    df = pd.read_parquet(path)
    df = df.dropna(subset=["qc (MPa)", "fs (MPa)", "Rf (%)"])   # u2 is no longer required

    df = df[(df["qc (MPa)"] > 0) & (df["fs (MPa)"] > 0)]
    df = df[(df["Rf (%)"] >= 0.001) & (df["Rf (%)"] <= 31)]
    df = df[(df["qc (MPa)"] < 200)]
    df = df[df["u2 (MPa)"].isna() | ((df["u2 (MPa)"] > 0) & (df["u2 (MPa)"] < 8))]

    
    return df.reset_index(drop=True)


def robertson_coordinates(df):
    """
    Convert CPT columns to Robertson chart coordinates.

    Missing u2 values are treated as 0 for the qt correction, so rows
    without pore pressure still plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'qc (MPa)', 'u2 (MPa)' and 'Rf (%)'.

    Returns
    -------
    x : numpy.ndarray
        log10 of Rf (%).
    y : numpy.ndarray
        log10 of qt / pa.
    """
    u2 = df["u2 (MPa)"].fillna(0).values
    qt_mpa = df["qc (MPa)"].values + u2 * (1 - NET_AREA_RATIO)
    x = np.log10(df["Rf (%)"].values)
    y = np.log10(qt_mpa * 1000 / PA_KPA)
    return x, y


#%% -----------------------------------------------------------------------------
# 3. Chart drawing and saving
# -----------------------------------------------------------------------------

def draw_robertson_chart(ax, image_path):
    """
    Draw the static Robertson (1986, updated 2010) chart on an axes.

    Adds the background image, log-style tick labels, minor ticks, grid,
    zone numbers and axis labels. Data are not drawn here.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    image_path : str
        Path to the Robertson chart background image.
    """
    background = image.imread(image_path)
    ax.imshow(background, extent=(*LOG_XLIM, *LOG_YLIM), zorder=0, aspect="auto")

    # Major ticks, labelled in real units
    ax.set_xticks(np.log10([0.1, 1, 10]))
    ax.set_xticklabels([0.1, 1, 10])
    ax.set_yticks(np.log10([1, 10, 100, 1000]))
    ax.set_yticklabels([1, 10, 100, 1000])
    ax.set_xlim(LOG_XLIM)
    ax.set_ylim(LOG_YLIM)

    # Minor ticks at 2..9 within each decade
    minor_x = np.log10(np.concatenate([np.arange(2, 10) / 10, np.arange(2, 10)]))
    minor_y = np.log10(np.concatenate([np.arange(2, 10) * base for base in (1, 10, 100)]))
    ax.set_xticks(minor_x, minor=True)
    ax.set_yticks(minor_y, minor=True)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)

    for x, y, label in CLUSTER_LABELS:
        ax.text(np.log10(x), np.log10(y), label, fontsize=10, color="k",
                va="center", ha="center")

    ax.grid(which="major", lw=0.1, c="grey", ls="dotted")
    ax.grid(which="minor", lw=0.1, c="grey", ls="dotted")
    ax.set_xlabel(r"$R_f~(\%)$", labelpad=3)
    ax.set_ylabel(r"$q_{t} / p_a~(-)$", labelpad=-5)


def new_chart(image_path):
    """
    Create a figure with the Robertson chart as background.

    Parameters
    ----------
    image_path : str
        Path to the Robertson chart background image.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Ready for data to be plotted on top.
    """
    fig, ax = plt.subplots(figsize=(FIG_SIZE_CM[0] * CM, FIG_SIZE_CM[1] * CM), dpi=200)
    draw_robertson_chart(ax, image_path)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.18)
    return fig, ax


def save_figure(fig, name):
    """
    Save a figure to SAVE_DIR as a PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File name without folder, e.g. "Dataset1_density_contour".
        The ".png" extension is added if missing.

    Returns
    -------
    str
        Full path of the saved file.
    """
    if not name.lower().endswith(".png"):
        name += ".png"
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


#%% -----------------------------------------------------------------------------
# 4. Density contour (KDE)
# -----------------------------------------------------------------------------

def density_contour(df, image_path, title, density_max, density_min=0.01,
                    level_step = 0.1, n_samples=N_SAMPLES, save_name=None, normalize = True):
    """
    Plot a filled density contour of CPT points on the Robertson chart.

    A random sample of up to `n_samples` points is used, because the KDE
    evaluation cost grows with the number of points. A Gaussian KDE is fitted
    in log space and evaluated on a 200 x 200 grid.

    How to read it: each contour level is a fixed density value. Density is
    probability per unit area in log space, so a value of 2 means that a
    small area dA around that point holds about 2 dA of the data. Regions
    with the highest levels are where the data are most concentrated.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `load_cpt_points`.
    image_path : str
        Path to the Robertson chart background image.
    title : str
        Plot title prefix, e.g. "Dataset 1".
    density_max : float
        Upper density level. Densities above it are filled with the top
        colour (extend="max"), so the high-density core is not left blank.
    density_min : float, default 0.01
        Lowest contour level. Must be greater than 0.
    level_step : float, default 0.2
        Step between contour levels. Levels run from density_min to
        density_max in this step, so the colorbar shows round numbers.
    n_samples : int, default N_SAMPLES
        Maximum number of points used to fit the KDE.
    save_name : str, optional
        If given, save the figure to SAVE_DIR under this name.
        If None (default), the figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    sample = df.sample(n=min(n_samples, len(df)), random_state=RANDOM_STATE)
    x_s, y_s = robertson_coordinates(sample)
    kde = gaussian_kde(np.vstack([x_s, y_s]))

    # Evaluate the density on a regular grid covering the chart
    xx, yy = np.meshgrid(np.linspace(*LOG_XLIM, 200), np.linspace(*LOG_YLIM, 200))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    if normalize:
        zz = zz / zz.max()

    # levels = np.linspace(density_min, density_max, n_levels)
    levels = np.round(np.arange(density_min, density_max + level_step / 2, level_step), 6)
    fig, ax = new_chart(image_path)
    filled = ax.contourf(xx, yy, zz, levels=levels, cmap=DENSITY_CMAP,
                         alpha=0.7, extend="neither" if normalize else "max", zorder=2)
    ax.contour(xx, yy, zz, levels=levels, colors="k", linewidths=0.4, zorder=3)
    # fig.colorbar(filled, ax=ax, pad=0.02, label="Density")
    # cbar = fig.colorbar(filled, ax=ax, pad=0.02, label="Density", ticks=levels)
    cbar = fig.colorbar(filled, ax=ax, pad=0.02,
                    label="Relative density (peak = 1)" if normalize else "Density",
                    ticks=levels)
    cbar.ax.set_yticklabels([f"{lvl:.1f}" for lvl in levels])
    # ax.set_title(f"{title} density contour, n = {len(sample)}", fontsize=9)
    ax.set_title(f"{title}", fontsize=9)

    if save_name is not None:
        save_figure(fig, save_name)
    return fig, ax


#%% -----------------------------------------------------------------------------
# 5. 2D density heatmap (histogram)
# -----------------------------------------------------------------------------

def density_heatmap(df, image_path, title, count_max=None, bins=40, save_name=None):
    """
    Plot a 2D histogram of all CPT points on the Robertson chart.

    Unlike the contour plot, this uses every point (no sampling). Empty bins
    are hidden so the chart background shows through.

    How to read it: each cell's colour is the number of points that fall
    inside it. Colour values are counts, not probabilities, so they depend on
    the total number of points.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `load_cpt_points`.
    image_path : str
        Path to the Robertson chart background image.
    title : str
        Plot title prefix, e.g. "Dataset 1".
    count_max : int, optional
        Upper end of the colour scale. Counts above it get the top colour.
        If None, the scale runs to the largest bin count.
    bins : int, default 40
        Number of bins along each axis.
    save_name : str, optional
        If given, save the figure to SAVE_DIR under this name.
        If None (default), the figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    x, y = robertson_coordinates(df)
    counts, _, _ = np.histogram2d(x, y, bins=bins, range=[LOG_XLIM, LOG_YLIM])

    # histogram2d returns [x, y]; imshow expects [y, x], so transpose.
    counts = np.ma.masked_equal(counts.T, 0)

    fig, ax = new_chart(image_path)
    # im = ax.imshow(counts, origin="lower", extent=(*LOG_XLIM, *LOG_YLIM),
    #                cmap=DENSITY_CMAP, aspect="auto", alpha=0.8, zorder=2,
    #                vmin=1, vmax=count_max)
    im = ax.imshow(counts, origin="lower", extent=(*LOG_XLIM, *LOG_YLIM),
               cmap=DENSITY_CMAP, aspect="auto", alpha=0.8, zorder=2,
               vmin=1, vmax=count_max)
    fig.colorbar(im, ax=ax, pad=0.02, label="Count per bin")
    ax.set_title(f"{title} 2D density heatmap, n = {len(df)}", fontsize=9)

    if save_name is not None:
        save_figure(fig, save_name)
    return fig, ax


#%% -----------------------------------------------------------------------------
# 6. Single-sounding plot (optional)
# -----------------------------------------------------------------------------

def get_random_id(df, id_col="ID", type_col="test type",
                  valid_types=("CPTu", "SCPTu")):
    """
    Pick a random sounding ID from the valid test types.

    Parameters
    ----------
    df : pandas.DataFrame
    id_col : str, default "ID"
    type_col : str, default "test type"
    valid_types : tuple of str, default ("CPTu", "SCPTu")

    Returns
    -------
    random_id : str or int
    subset : pandas.DataFrame
        All rows belonging to that sounding.
    """
    valid_ids = df.loc[df[type_col].isin(valid_types), id_col].unique()
    if len(valid_ids) == 0:
        raise ValueError("No IDs found with the given test types.")
    random_id = np.random.choice(valid_ids)
    return random_id, df[df[id_col] == random_id]


def add_metadata_footnote(fig, df, id_col="ID", db_col="database"):
    """
    Add a footnote listing the sounding ID(s), data source and pa.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    df : pandas.DataFrame
    id_col, db_col : str
        Column names for the sounding ID and data source. If a column is
        missing, "n/a" is written instead.
    """
    id_str = ", ".join(map(str, df[id_col].unique())) if id_col in df.columns else "n/a"
    db_str = ", ".join(map(str, df[db_col].unique())) if db_col in df.columns else "n/a"
    footnote = (
        f"ID: {id_str}   |   Database: {db_str}\n"
        r"$p_{a}$: atmospheric pressure (101.3 kPa)"
    )
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=9)


def plot_robertson_with_depth(image_path, df, depth_column="Depth (m)",
                              save_path=None):
    """
    Plot one or more soundings on the Robertson chart, coloured by depth.

    Parameters
    ----------
    image_path : str
        Path to the Robertson chart background image.
    df : pandas.DataFrame
        Must contain 'qc (MPa)', 'u2 (MPa)', 'Rf (%)' and `depth_column`.
    depth_column : str, default "Depth (m)"
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    x, y = robertson_coordinates(df)
    depth = df[depth_column].values

    fig, ax = new_chart(image_path)
    points = ax.scatter(x, y, c=depth, cmap="viridis", s=5,
                        edgecolors="none", alpha=0.8, zorder=2)
    cbar = fig.colorbar(points, ax=ax, pad=0.02)
    cbar.set_label(depth_column, labelpad=5)
    cbar.ax.invert_yaxis()  # depth increases downward

    add_metadata_footnote(fig, df)

    if save_path is not None:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight")
    return fig, ax


#%% -----------------------------------------------------------------------------
# 7. Run: Dataset 1
# -----------------------------------------------------------------------------

df1 = load_cpt_points(DATASET1_PATH)

density_contour(df1, IMAGE_PATH, "Dataset 1", density_max=1.0,
                save_name="Figure14_a_Dataset1_density_contour")
density_heatmap(df1, IMAGE_PATH, "Dataset 1", count_max=25000,
                save_name="Figure14_b_Dataset1_density_heatmap")

#%% -----------------------------------------------------------------------------
# 8. Run: Dataset 2
# -----------------------------------------------------------------------------

df2 = load_cpt_points(DATASET2_PATH)

density_contour(df2, IMAGE_PATH, "Dataset 2", density_max=1.0,
                save_name="Figure14_c_Dataset2_density_contour")
density_heatmap(df2, IMAGE_PATH, "Dataset 2",
                save_name="Figure14_d_Dataset2_density_heatmap")

# %%

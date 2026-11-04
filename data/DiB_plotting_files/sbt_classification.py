"""
sbt_classification.py

Core Robertson (2010) SBT classification logic, extracted from
cpt_processing.py with all GEF/file-parsing machinery removed
(parse_gef, parse_plain_cpt, ParsedCPT, column inference, etc.) — this
version takes depth/qc/fs/u2 directly as arrays or a DataFrame, since the
data already lives in a spreadsheet, not a .gef file.

DEPENDENCY: classify_robertson_2010() looks for 30 boundary-curve CSVs
(Curve1.csv ... Curve30.csv) in a `sbt_chart/` folder next to this file —
same as the original script. Without that folder, every record classifies
as zone 0 ("Unclassified") rather than failing. Copy your sbt_chart/
folder alongside this file for real zone numbers.

IMPORTANT: the actual zone lookup (compute_sbt_zone) needs ONLY qc, fs,
and u2 — no depth, no groundwater depth, no assumed unit weight, no
effective stress. classify_robertson_2010() is called with qt (normalized
only by atmospheric pressure, inside that function) and Rf, not with the
stress-normalized Qtn. Depth/gwt/gamma only matter for the separate,
optional compute_stress_dependent_extras() (Qtn, Ic, alpha_m, the
Robertson constrained-modulus M) — call that only if you actually want
those columns, since they're not part of the classification.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

GAMMA_W = 9.81
PA_MPA = 0.1013

SBT_ZONES_2010 = [
    {"zone": 1, "name": "Sensitive fine-grained", "color": "#94a3b8"},
    {"zone": 2, "name": "Organic soils - clay", "color": "#4ade80"},
    {"zone": 3, "name": "Clay", "color": "#60a5fa"},
    {"zone": 4, "name": "Silt mixtures", "color": "#818cf8"},
    {"zone": 5, "name": "Sand mixtures", "color": "#fbbf24"},
    {"zone": 6, "name": "Sands", "color": "#fb923c"},
    {"zone": 7, "name": "Gravelly sand to dense sand", "color": "#f97316"},
    {"zone": 8, "name": "Very stiff sand to clayey sand*", "color": "#ef4444"},
    {"zone": 9, "name": "Very stiff fine-grained*", "color": "#dc2626"},
]


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _load_robertson_2010_polygons(chart_dir: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    if not chart_dir.is_dir():
        return []

    curves: dict[str, list[tuple[float, float]]] = {}
    for i in range(1, 31):
        csv_path = chart_dir / f"Curve{i}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=";")
            curves[f"C{i}"] = list(zip(df["x"].tolist(), df["y"].tolist()))
    for tag in ("C4_P5", "C10_P4", "C26_P8"):
        fname = tag.replace("C", "Curve", 1) + ".csv"
        csv_path = chart_dir / fname
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=";")
            curves[tag] = list(zip(df["x"].tolist(), df["y"].tolist()))

    def _cat(*keys: str) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for k in keys:
            pts.extend(curves.get(k, []))
        return pts

    return [
        (1, _cat("C1", "C2", "C3", "C4", "C5")),
        (2, _cat("C6", "C7", "C8")),
        (3, _cat("C8", "C2", "C10", "C12", "C11")),
        (4, _cat("C14", "C10_P4", "C3", "C13")),
        (5, _cat("C13", "C25", "C24", "C23", "C4_P5")),
        (6, _cat("C24", "C22", "C27", "C19", "C26")),
        (7, _cat("C20", "C21", "C27")),
        (8, _cat("C25", "C26_P8", "C18", "C15")),
        (9, _cat("C15", "C14", "C12", "C16", "C17")),
        (2, _cat("C28", "C29", "C30")),
    ]


_ROBERTSON_2010_POLYGONS: list[tuple[int, list[tuple[float, float]]]] | None = None
_CHART_DIR_USED: Path | None = None


def classify_robertson_2010(rf_percent: float, qt_mpa: float, chart_dir: Path) -> dict:
    global _ROBERTSON_2010_POLYGONS, _CHART_DIR_USED
    if _ROBERTSON_2010_POLYGONS is None or _CHART_DIR_USED != chart_dir:
        _ROBERTSON_2010_POLYGONS = _load_robertson_2010_polygons(chart_dir)
        _CHART_DIR_USED = chart_dir

    if not _ROBERTSON_2010_POLYGONS:
        return _sbt2010_zone_dict(0)

    rf_clamped = max(rf_percent, 0.1)
    qt_over_pa = qt_mpa / PA_MPA
    for zone_num, poly in _ROBERTSON_2010_POLYGONS:
        if _point_in_polygon(rf_clamped, qt_over_pa, poly):
            return _sbt2010_zone_dict(zone_num)
    return _sbt2010_zone_dict(0)


def _sbt2010_zone_dict(zone: int) -> dict:
    for z in SBT_ZONES_2010:
        if z["zone"] == zone:
            return z
    return {"zone": 0, "name": "Unclassified", "color": "#e5e7eb"}


def robertson_alpha_m(ic: float, qt: float) -> float:
    if ic <= 2.2:
        return 0.0188 * 10 ** (0.55 * ic + 1.68)
    return min(max(qt, 0.0), 14.0)


def compute_sbt_zone(
    qc_mpa: np.ndarray,
    fs_mpa: np.ndarray,
    u2_mpa: np.ndarray,
    chart_dir: Path,
    a_net: float = 0.8,
) -> pd.DataFrame:
    """SBT zone classification only. Needs qc, fs, u2 (all MPa) and the net
    area ratio a_net — nothing else. No depth, no groundwater depth, no
    assumed unit weight, no effective stress: the actual Robertson (2010)
    zone lookup here is done on qt (normalized only by atmospheric
    pressure, inside classify_robertson_2010) and Rf, not on the
    stress-normalized Qtn. Those stress-dependent quantities (Qtn, Ic,
    alpha_m, M) are a separate, optional calculation — see
    compute_stress_dependent_extras() — used for constrained-modulus
    estimation, not for the zone number itself.
    """
    qc = np.asarray(qc_mpa, dtype=float)
    fs = np.asarray(fs_mpa, dtype=float)
    u2 = np.asarray(u2_mpa, dtype=float)

    qt = qc + u2 * (1.0 - a_net)
    rf = np.where(qt > 0, fs / qt * 100.0, np.nan)

    sbt = [classify_robertson_2010(rf_val, qt_val, chart_dir) for rf_val, qt_val in zip(rf, qt)]

    return pd.DataFrame({
        "qt_mpa": qt,
        "rf_percent": rf,
        "sbt_zone_computed": [z["zone"] for z in sbt],
        "sbt_name_computed": [z["name"] for z in sbt],
    })


def compute_stress_dependent_extras(
    depth_m: np.ndarray,
    qt_mpa: np.ndarray,
    fs_mpa: np.ndarray,
    gwt: float = 2.0,
    gamma: float = 18.0,
) -> pd.DataFrame:
    """OPTIONAL extras that DO require a stress assumption: Qtn (stress-
    normalized cone resistance), Ic (soil behaviour type index), and the
    Robertson constrained-modulus estimate M. Not needed for sbt_zone —
    only compute this if you actually want those columns, since it
    introduces the gwt/gamma assumptions that the classification itself
    doesn't need.

    sigma_v uses a single CONSTANT assumed unit weight (gamma) for the
    whole profile, so it collapses to sigma_v = gamma * depth per row —
    see the module docstring for why per-sounding grouping isn't needed.
    """
    depth = np.asarray(depth_m, dtype=float)
    qt = np.asarray(qt_mpa, dtype=float)
    fs = np.asarray(fs_mpa, dtype=float)

    sigma_v_kpa = gamma * depth
    u0_kpa = np.where(depth > gwt, GAMMA_W * (depth - gwt), 0.0)
    sigma_v_eff_kpa = np.maximum(sigma_v_kpa - u0_kpa, 0.0)

    qt_net_mpa = qt - sigma_v_kpa / 1000.0
    qt_net_positive = np.maximum(qt_net_mpa, 0.001)

    qt_norm = np.where(qt_net_mpa > 0, qt_net_mpa / np.maximum(sigma_v_eff_kpa / 1000.0, 1e-6), 0.1)
    fr = np.where(qt_net_mpa > 0, fs / qt_net_positive * 100.0, 0.01)
    ic = np.sqrt((3.47 - np.log10(np.maximum(qt_norm, 0.1))) ** 2 + (np.log10(np.maximum(fr, 0.1)) + 1.22) ** 2)
    alpha_m = np.array([robertson_alpha_m(ic_v, qt_v) for ic_v, qt_v in zip(ic, qt_norm)])

    return pd.DataFrame({
        "sigma_v_kpa": sigma_v_kpa,
        "sigma_v_eff_kpa": sigma_v_eff_kpa,
        "u0_kpa": u0_kpa,
        "Qtn": qt_norm,
        "Fr_percent": fr,
        "Ic": ic,
        "alpha_m_robertson": alpha_m,
        "M_robertson_mpa": alpha_m * qt_net_positive,
    })
from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass
class ParsedCPT:
    path: Path
    header: dict[str, str]
    rows: pd.DataFrame


def read_text_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin1", errors="replace")


def parse_float(value: str | None, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return default


def parse_gef(path: Path) -> ParsedCPT:
    text = read_text_fallback(path)
    lines = text.splitlines()
    header: dict[str, str] = {}
    column_map: dict[int, int] = {}
    separator: str | None = None
    void_values: dict[int, float] = {}
    global_void: float | None = None
    data_start = 0
    zid: float | None = None

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        upper = line.upper()
        if upper.startswith("#EOH"):
            data_start = i + 1
            break
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip()

        if key == "#COLUMNSEPARATOR":
            separator = value or separator
        elif key == "#COLUMNINFO":
            parts = split_header_values(value)
            if len(parts) >= 4:
                col_idx = int(parse_float(parts[0], 0)) - 1
                quantity_code = int(parse_float(parts[-1], -1))
                column_map[col_idx] = quantity_code
        elif key == "#COLUMNVOID":
            parts = split_header_values(value)
            if len(parts) >= 2:
                col_idx = int(parse_float(parts[0], 0)) - 1
                void_values[col_idx] = parse_float(parts[1])
            elif parts:
                global_void = parse_float(parts[0])
        elif key == "#ZID":
            parts = split_header_values(value)
            if len(parts) >= 2:
                zid = parse_float(parts[1])
        elif key == "#MEASUREMENTVAR":
            parts = [part.strip() for part in value.split(",")]
            if len(parts) >= 2:
                var_id = int(parse_float(parts[0], -1))
                var_value = parse_float(parts[1])
                if var_id == 3 and np.isfinite(var_value):
                    header["a_net"] = str(var_value)
                elif var_id == 14 and np.isfinite(var_value):
                    header["gwt"] = str(abs(var_value))
        elif key in {"#TESTID", "#STARTDATE", "#PROJECTNAME", "#LOCATIONNAME", "#CLIENTNAME"}:
            header[key.lstrip("#").lower()] = value

    col_depth = first_column_with_code(column_map, {1, 10})
    col_qc = first_column_with_code(column_map, {2})
    col_fs = first_column_with_code(column_map, {3})
    col_u2 = first_column_with_code(column_map, {5, 6})

    sample_values = collect_sample_values(lines[data_start:], separator, void_values, global_void)
    col_depth, col_qc, col_fs, col_u2 = infer_missing_columns(
        sample_values=sample_values,
        col_depth=col_depth,
        col_qc=col_qc,
        col_fs=col_fs,
        col_u2=col_u2,
    )

    parsed_rows = []
    for raw_line in lines[data_start:]:
        line = raw_line.strip()
        if not line or line.startswith(("!", "#")):
            continue
        parts = split_data_line(line, separator)
        values = [parse_float(part) for part in parts]
        if len(values) < 2:
            continue

        if has_void(values, void_values, global_void):
            continue

        depth = get_column(values, col_depth)
        qc = get_column(values, col_qc)
        fs = get_column(values, col_fs, 0.0)
        u2 = get_column(values, col_u2, 0.0)
        if not np.isfinite(depth) or not np.isfinite(qc) or qc <= 0:
            continue
        if zid is not None and depth < 0:
            depth = zid - depth
        if depth < 0:
            continue

        qc = qc / 1000.0 if qc > 100 else qc
        fs = fs / 1000.0 if abs(fs) > 10 else fs
        u2 = u2 / 1000.0 if abs(u2) > 100 else u2
        parsed_rows.append({"depth": depth, "qc": qc, "fs": fs, "u2": u2})

    rows = make_rows_frame(parsed_rows)
    return ParsedCPT(path=path, header=header, rows=rows)


def parse_plain_cpt(path: Path) -> ParsedCPT:
    text = read_text_fallback(path)
    parsed_rows = []
    for raw_line in text.splitlines():
        parts = raw_line.strip().replace(",", ".").split()
        if len(parts) < 3:
            parts = [part for part in raw_line.replace(",", ".").replace(";", " ").split()]
        values = [parse_float(part) for part in parts]
        if len(values) < 3 or not all(np.isfinite(values[:3])):
            continue
        depth, qc, fs = values[:3]
        u2 = values[3] if len(values) > 3 and np.isfinite(values[3]) else 0.0
        if depth < 0 or qc <= 0:
            continue
        parsed_rows.append(
            {
                "depth": depth,
                "qc": qc / 1000.0 if qc > 100 else qc,
                "fs": fs / 1000.0 if abs(fs) > 10 else fs,
                "u2": u2 / 1000.0 if abs(u2) > 100 else u2,
            }
        )
    rows = make_rows_frame(parsed_rows)
    return ParsedCPT(path=path, header={}, rows=rows)


def parse_cpt_file(path: Path) -> ParsedCPT:
    suffix = path.suffix.lower()
    if suffix == ".gef":
        return parse_gef(path)
    return parse_plain_cpt(path)


def first_column_with_code(column_map: dict[int, int], codes: set[int]) -> int | None:
    for column, code in column_map.items():
        if code in codes:
            return column
    return None


def split_header_values(value: str) -> list[str]:
    return [part for part in re.split(r"[\s,;]+", value.strip()) if part]


def split_data_line(line: str, separator: str | None) -> list[str]:
    if separator == ";" and ";" in line:
        return [part.strip() for part in line.split(";")]
    if separator == "," and "," in line:
        return [part.strip() for part in line.split(",")]
    return line.split()


def collect_sample_values(
    data_lines: list[str],
    separator: str | None,
    void_values: dict[int, float],
    global_void: float | None,
    limit: int = 200,
) -> list[list[float]]:
    sample_values = []
    for raw_line in data_lines:
        line = raw_line.strip()
        if not line or line.startswith(("!", "#")):
            continue
        values = [parse_float(part) for part in split_data_line(line, separator)]
        if len(values) < 2 or has_void(values, void_values, global_void):
            continue
        if sum(np.isfinite(value) for value in values) >= 2:
            sample_values.append(values)
        if len(sample_values) >= limit:
            break
    return sample_values


def infer_missing_columns(
    sample_values: list[list[float]],
    col_depth: int | None,
    col_qc: int | None,
    col_fs: int | None,
    col_u2: int | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    if col_depth is not None and col_qc is not None:
        return col_depth, col_qc, col_fs, col_u2
    if not sample_values:
        return col_depth, col_qc, col_fs, col_u2

    max_cols = max(len(values) for values in sample_values)
    columns = []
    for col_idx in range(max_cols):
        column_values = np.array(
            [values[col_idx] for values in sample_values if col_idx < len(values) and np.isfinite(values[col_idx])],
            dtype=float,
        )
        if len(column_values) < 5:
            continue
        diffs = np.diff(column_values)
        monotonic_score = float(np.mean(diffs >= -1e-9)) if len(diffs) else 0.0
        columns.append(
            {
                "idx": col_idx,
                "min": float(np.min(column_values)),
                "max": float(np.max(column_values)),
                "median": float(np.median(column_values)),
                "monotonic_score": monotonic_score,
            }
        )

    if col_depth is None:
        depth_candidates = [
            col
            for col in columns
            if col["min"] >= -0.01 and col["max"] <= 200 and col["monotonic_score"] > 0.95
        ]
        if depth_candidates:
            col_depth = sorted(depth_candidates, key=lambda col: (col["idx"] != 0, col["max"]))[0]["idx"]

    remaining = [col for col in columns if col["idx"] != col_depth]
    if col_qc is None:
        qc_candidates = [col for col in remaining if col["max"] > 0 and col["median"] > 0]
        if qc_candidates:
            col_qc = sorted(qc_candidates, key=lambda col: (-col["max"], col["idx"]))[0]["idx"]

    remaining = [col for col in remaining if col["idx"] != col_qc]
    if col_fs is None and remaining:
        positive_small = [col for col in remaining if col["median"] >= 0]
        if positive_small:
            col_fs = sorted(positive_small, key=lambda col: (abs(col["median"]), col["idx"]))[0]["idx"]

    remaining = [col for col in remaining if col["idx"] != col_fs]
    if col_u2 is None and remaining:
        col_u2 = remaining[0]["idx"]

    return col_depth, col_qc, col_fs, col_u2


def has_void(values: list[float], void_values: dict[int, float], global_void: float | None) -> bool:
    for column, void_value in void_values.items():
        if column < len(values) and np.isfinite(void_value) and values[column] == void_value:
            return True
    if global_void is not None and np.isfinite(global_void):
        return any(value == global_void for value in values)
    return False


def get_column(values: list[float], column: int | None, default: float = np.nan) -> float:
    if column is None or column >= len(values):
        return default
    return values[column]


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


def _load_robertson_2010_polygons() -> list[tuple[int, list[tuple[float, float]]]]:
    chart_dir = Path(__file__).parent / "sbt_chart"
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


def classify_robertson_2010(rf_percent: float, qt_mpa: float) -> dict:
    global _ROBERTSON_2010_POLYGONS
    if _ROBERTSON_2010_POLYGONS is None:
        _ROBERTSON_2010_POLYGONS = _load_robertson_2010_polygons()

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


def make_rows_frame(parsed_rows: list[dict[str, float]]) -> pd.DataFrame:
    rows = pd.DataFrame(parsed_rows, columns=["depth", "qc", "fs", "u2"])
    if rows.empty:
        return rows
    return rows.sort_values("depth").reset_index(drop=True)


def compute_cpt(
    rows: pd.DataFrame,
    gwt: float = 2.0,
    gamma: float = 18.0,
    a_net: float = 0.8,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    data = rows.copy()
    data["depth"] = pd.to_numeric(data["depth"], errors="coerce")
    data["qc"] = pd.to_numeric(data["qc"], errors="coerce")
    data["fs"] = pd.to_numeric(data["fs"], errors="coerce").fillna(0.0)
    data["u2"] = pd.to_numeric(data["u2"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["depth", "qc"]).sort_values("depth").reset_index(drop=True)

    depth = data["depth"].to_numpy(dtype=float)
    dz = np.diff(np.r_[0.0, depth])
    dz = np.maximum(dz, 0.0)
    sigma_v_kpa = np.cumsum(gamma * dz)
    u0_kpa = np.where(depth > gwt, GAMMA_W * (depth - gwt), 0.0)
    sigma_v_eff_kpa = np.maximum(sigma_v_kpa - u0_kpa, 0.0)

    qc = data["qc"].to_numpy(dtype=float)
    fs = data["fs"].to_numpy(dtype=float)
    u2 = data["u2"].to_numpy(dtype=float)
    qt = qc + u2 * (1.0 - a_net)
    qt_net_mpa = qt - sigma_v_kpa / 1000.0
    qt_net_positive = np.maximum(qt_net_mpa, 0.001)

    rf = np.where(qt > 0, fs / qt * 100.0, np.nan)
    qt_norm = np.where(qt_net_mpa > 0, qt_net_mpa / np.maximum(sigma_v_eff_kpa / 1000.0, 1e-6), 0.1)
    fr = np.where(qt_net_mpa > 0, fs / qt_net_positive * 100.0, 0.01)

    ic = np.sqrt((3.47 - np.log10(np.maximum(qt_norm, 0.1))) ** 2 + (np.log10(np.maximum(fr, 0.1)) + 1.22) ** 2)

    sbt_2010 = [classify_robertson_2010(rf_val, qt_val) for rf_val, qt_val in zip(rf, qt)]
    alpha_m = np.array([robertson_alpha_m(ic_value, qt_value) for ic_value, qt_value in zip(ic, qt_norm)])

    data["sigma_v_kpa"] = sigma_v_kpa
    data["sigma_v_eff_kpa"] = sigma_v_eff_kpa
    data["u0_kpa"] = u0_kpa
    data["qt_mpa"] = qt
    data["fs_kpa"] = fs * 1000.0
    data["u2_kpa"] = u2 * 1000.0
    data["rf_percent"] = rf
    data["Qt"] = qt_norm
    data["Fr_percent"] = fr
    data["Ic"] = ic
    data["sbt_zone"] = [zone["zone"] for zone in sbt_2010]
    data["sbt_name"] = [zone["name"] for zone in sbt_2010]
    data["alpha_m_robertson"] = alpha_m
    data["M_robertson_mpa"] = alpha_m * qt_net_positive
    return data


def robertson_alpha_m(ic: float, qt: float) -> float:
    if ic <= 2.2:
        return 0.0188 * 10 ** (0.55 * ic + 1.68)
    return min(max(qt, 0.0), 14.0)


def discover_cpt_files(input_dir: Path, include_dissipation: bool = True) -> list[Path]:
    patterns = ("*.gef", "*.GEF", "*.cpt", "*.CPT", "*.txt", "*.TXT")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(input_dir.glob(pattern))
    unique_paths = sorted(set(paths), key=lambda path: path.name.lower())
    if include_dissipation:
        return unique_paths
    return [path for path in unique_paths if "_D" not in path.stem.upper()]

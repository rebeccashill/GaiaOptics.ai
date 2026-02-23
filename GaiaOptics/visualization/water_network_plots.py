# gaiaoptics/visualization/water_network_plots.py
from __future__ import annotations

"""
Water network plotting utilities (Phase 3 polish).

Goal:
- Produce 1 simple, credible plot:
    tank level over time
- Read outputs/<scenario>/traces.csv and write outputs/<scenario>/plots/tank_level_over_time.png

Defensive behavior:
- Tries multiple common column names for time and tank level.
- Raises a clear error if required columns are missing.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


_TIME_CANDIDATES = ("t", "time", "step", "hour")
_LEVEL_CANDIDATES = (
    "tank_level_m3",
    "tank_m3",
    "tank_level",
    "tank_volume_m3",
    "level_m3",
)


def _read_traces_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"traces.csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"traces.csv has no header: {path}")
        rows = [row for row in reader]
        return list(reader.fieldnames), rows


def _pick_column(header: List[str], candidates: Tuple[str, ...]) -> str:
    header_set = {h.strip() for h in header}

    # exact match first
    for c in candidates:
        if c in header_set:
            return c

    # case-insensitive match
    lowered = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]

    raise KeyError(f"missing column; tried: {list(candidates)}; available: {header}")


def _to_float_list(rows: List[Dict[str, str]], key: str) -> List[float]:
    out: List[float] = []
    for i, r in enumerate(rows):
        v = r.get(key, "")
        try:
            out.append(float(v))
        except Exception as e:
            raise ValueError(f"could not parse float at row {i} column '{key}': {v!r}") from e
    return out


def plot_tank_level_over_time(
    traces_csv: Path,
    out_png: Path,
    *,
    title: str | None = None,
) -> Path:
    """
    Reads traces_csv and writes a single PNG to out_png.
    Returns out_png for convenience.
    """
    header, rows = _read_traces_csv(traces_csv)
    if not rows:
        raise ValueError(f"traces.csv has no rows: {traces_csv}")

    time_col = _pick_column(header, _TIME_CANDIDATES)
    level_col = _pick_column(header, _LEVEL_CANDIDATES)

    t = _to_float_list(rows, time_col)
    level = _to_float_list(rows, level_col)

    plt.figure()
    plt.plot(t, level)
    plt.xlabel(time_col)
    plt.ylabel(level_col)
    plt.title(title or "Water network tank level over time")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

    return out_png


def plot_water_network_outputs_dir(outputs_dir: Path) -> Path:
    """
    Convenience helper:
      outputs/<scenario>/traces.csv -> outputs/<scenario>/plots/tank_level_over_time.png
    """
    traces_csv = outputs_dir / "traces.csv"
    out_png = outputs_dir / "plots" / "tank_level_over_time.png"
    return plot_tank_level_over_time(
        traces_csv,
        out_png,
        title=f"Water network tank level over time — {outputs_dir.name}",
    )
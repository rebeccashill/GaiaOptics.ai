# gaiaoptics/domains/data_center/objective.py
"""
Data center objective (Phase 2).

Simple score (lower is better):
  score = energy_cost + emissions + unmet_work_penalty + soft_constraint_penalty

Where:
- energy_cost = sum_t (grid_import_kW[t] * dt_hours * price_per_kWh[t])
- emissions   = sum_t (grid_import_kW[t] * dt_hours * carbon_kg_per_kWh[t])

Optional terms:
- unmet_work_penalty: if workload deferral is enabled and some work is deferred,
  penalize ending backlog (or total deferred energy) to represent SLA pressure.
  In Phase 2, default penalty is light but nonzero to avoid "turn everything off".

- soft_constraint_penalty: generic penalty for SOFT constraints if present later.

This mirrors the microgrid pattern so reports stay uniform across domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from gaiaoptics.core.types import ConstraintResult, ObjectiveResult, Severity


def _series(traces: Dict[str, Any], key: str, n_steps: int) -> List[float]:
    v = traces.get(key)
    if not isinstance(v, list) or len(v) != n_steps or not all(isinstance(x, (int, float)) for x in v):
        raise ValueError(f"simulate() must return '{key}' as list[float] with length n_steps")
    return [float(x) for x in v]


def _series_or_scalar(cfg: Dict[str, Any], key: str, n_steps: int, default_scalar: float) -> List[float]:
    v = cfg.get(key, None)
    if v is None:
        return [float(default_scalar)] * n_steps
    if isinstance(v, (int, float)):
        return [float(v)] * n_steps
    if isinstance(v, list) and len(v) == n_steps and all(isinstance(x, (int, float)) for x in v):
        return [float(x) for x in v]
    raise ValueError(f"'{key}' must be a number or a list of {n_steps} numbers")


@dataclass(frozen=True)
class DataCenterObjectiveConfig:
    n_steps: int
    dt_hours: float

    # Series (len n_steps) or scalars
    price_per_kwh: List[float]
    carbon_kg_per_kwh: List[float]

    # Optional SLA-ish penalty (Phase 2)
    backlog_end_penalty_per_kwh: float = 5.0

    # Generic soft constraint penalty weight
    soft_penalty_weight: float = 1000.0


def compute_data_center_objective(
    traces: Dict[str, Any],
    constraints: Sequence[ConstraintResult],
    cfg: DataCenterObjectiveConfig,
) -> ObjectiveResult:
    n = int(cfg.n_steps)
    dt = float(cfg.dt_hours)

    grid_import_kw = _series(traces, "grid_import_kw", n)

    # --- core terms ---
    energy_cost = 0.0
    emissions_kg = 0.0
    for t in range(n):
        e_kwh = float(grid_import_kw[t]) * dt
        energy_cost += e_kwh * float(cfg.price_per_kwh[t])
        emissions_kg += e_kwh * float(cfg.carbon_kg_per_kwh[t])

    # --- backlog end penalty (optional) ---
    backlog_end_kwh = 0.0
    if "backlog_kwh" in traces:
        try:
            backlog_series = _series(traces, "backlog_kwh", n)
            backlog_end_kwh = float(backlog_series[-1]) if backlog_series else 0.0
        except Exception:
            backlog_end_kwh = 0.0

    backlog_penalty = backlog_end_kwh * float(cfg.backlog_end_penalty_per_kwh)

    # --- generic soft constraint penalty ---
    soft_pen_raw = 0.0
    for c in constraints:
        if c.severity == Severity.SOFT and c.margin < 0.0:
            soft_pen_raw += (-c.margin)

    soft_penalty = float(cfg.soft_penalty_weight) * float(soft_pen_raw)

    score = float(energy_cost + emissions_kg + backlog_penalty + soft_penalty)

    return ObjectiveResult(
        score=score,
        components={
            "energy_cost": float(energy_cost),
            "emissions_kg": float(emissions_kg),
            "backlog_end_kwh": float(backlog_end_kwh),
            "backlog_penalty": float(backlog_penalty),
            "soft_penalty_raw": float(soft_pen_raw),
            "soft_penalty": float(soft_penalty),
        },
        metadata={
            "score_convention": "lower_is_better",
            "weights": {
                "cost": 1.0,
                "emissions": 1.0,
                "backlog_end_penalty_per_kwh": float(cfg.backlog_end_penalty_per_kwh),
                "soft_penalty_weight": float(cfg.soft_penalty_weight),
            },
        },
    )


def objective_config_from_yaml(cfg: Dict[str, Any], n_steps: int, dt_hours: float) -> DataCenterObjectiveConfig:
    """
    Parses objective inputs from the scenario YAML dict.

    Looks in:
      cfg["series"]["price_per_kwh"] (scalar or list)
      cfg["series"]["carbon_kg_per_kwh"] (scalar or list)
      cfg["penalties"]["backlog_end_penalty_per_kwh"]
      cfg["penalties"]["soft_penalty_weight"]
    """
    series = cfg.get("series", {}) or {}
    penalties = cfg.get("penalties", {}) or {}

    price = _series_or_scalar(series, "price_per_kwh", n_steps, default_scalar=0.2)
    carbon = _series_or_scalar(series, "carbon_kg_per_kwh", n_steps, default_scalar=0.4)

    return DataCenterObjectiveConfig(
        n_steps=int(n_steps),
        dt_hours=float(dt_hours),
        price_per_kwh=price,
        carbon_kg_per_kwh=carbon,
        backlog_end_penalty_per_kwh=float(penalties.get("backlog_end_penalty_per_kwh", 5.0)),
        soft_penalty_weight=float(penalties.get("soft_penalty_weight", 1000.0)),
    )
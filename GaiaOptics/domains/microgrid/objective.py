# gaiaoptics/domains/microgrid/objective.py
"""
Microgrid objective.

Simple score (lower is better):
  score = energy_cost + emissions + unmet_load_penalty + soft_constraint_penalty

Where:
- energy_cost = sum_t (grid_import_kW[t] * dt_hours * price_per_kWh[t])
- emissions   = sum_t (grid_import_kW[t] * dt_hours * carbon_kg_per_kWh[t])
- unmet_load_penalty = unserved_kWh * unserved_energy_per_kWh (if unserved modeled)

Notes:
- In the starter "unconstrained grid" model, unmet load is typically 0 unless
  you explicitly cap grid import and compute unserved from net demand.
- SOFT constraint penalties are supported generically if you later add SOFT constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
class MicrogridObjectiveConfig:
    n_steps: int
    dt_hours: float

    # Series (len n_steps) or scalars
    price_per_kwh: List[float]
    carbon_kg_per_kwh: List[float]

    # Penalties / weights
    unserved_energy_per_kwh: float = 1000.0
    soft_penalty_weight: float = 1000.0


def compute_microgrid_objective(
    traces: Dict[str, Any],
    constraints: Sequence[ConstraintResult],
    cfg: MicrogridObjectiveConfig,
) -> ObjectiveResult:
    n = cfg.n_steps
    dt = float(cfg.dt_hours)

    grid_import_kw = _series(traces, "grid_import_kw", n)

    # Optional: if you model unserved explicitly in traces
    unserved_kw: Optional[List[float]] = None
    if "unserved_kw" in traces:
        try:
            unserved_kw = _series(traces, "unserved_kw", n)
        except Exception:
            unserved_kw = None

    # --- core terms ---
    energy_cost = 0.0
    emissions_kg = 0.0
    for t in range(n):
        e_kwh = float(grid_import_kw[t]) * dt
        energy_cost += e_kwh * float(cfg.price_per_kwh[t])
        emissions_kg += e_kwh * float(cfg.carbon_kg_per_kwh[t])

    # --- unmet load ---
    unserved_kwh = 0.0
    if unserved_kw is not None:
        unserved_kwh = sum(float(x) * dt for x in unserved_kw)

    unmet_load_penalty = float(unserved_kwh) * float(cfg.unserved_energy_per_kwh)

    # --- generic soft constraint penalty ---
    soft_pen = 0.0
    for c in constraints:
        if c.severity == Severity.SOFT and c.margin < 0.0:
            soft_pen += (-c.margin)

    soft_penalty = float(cfg.soft_penalty_weight) * float(soft_pen)

    score = float(energy_cost + emissions_kg + unmet_load_penalty + soft_penalty)

    return ObjectiveResult(
        score=score,
        components={
            "energy_cost": float(energy_cost),
            "emissions_kg": float(emissions_kg),
            "unserved_kwh": float(unserved_kwh),
            "unmet_load_penalty": float(unmet_load_penalty),
            "soft_penalty_raw": float(soft_pen),
            "soft_penalty": float(soft_penalty),
        },
        metadata={
            "score_convention": "lower_is_better",
            "weights": {
                "cost": 1.0,
                "emissions": 1.0,
                "unserved_energy_per_kwh": float(cfg.unserved_energy_per_kwh),
                "soft_penalty_weight": float(cfg.soft_penalty_weight),
            },
        },
    )


def objective_config_from_yaml(cfg: Dict[str, Any], n_steps: int, dt_hours: float) -> MicrogridObjectiveConfig:
    """
    Parses objective inputs from the scenario YAML dict.

    Looks in:
      cfg["series"]["price_per_kwh"] (scalar or list)
      cfg["series"]["carbon_kg_per_kwh"] (scalar or list)
      cfg["penalties"]["unserved_energy_per_kwh"]
      cfg["penalties"]["soft_penalty_weight"]
    """
    series = cfg.get("series", {}) or {}
    penalties = cfg.get("penalties", {}) or {}

    price = _series_or_scalar(series, "price_per_kwh", n_steps, default_scalar=0.2)
    carbon = _series_or_scalar(series, "carbon_kg_per_kwh", n_steps, default_scalar=0.4)

    return MicrogridObjectiveConfig(
        n_steps=n_steps,
        dt_hours=float(dt_hours),
        price_per_kwh=price,
        carbon_kg_per_kwh=carbon,
        unserved_energy_per_kwh=float(penalties.get("unserved_energy_per_kwh", 1000.0)),
        soft_penalty_weight=float(penalties.get("soft_penalty_weight", 1000.0)),
    )
# gaiaoptics/domains/microgrid/mission.py
"""
Microgrid domain entrypoint.

Defines build_problem_from_config(cfg) -> Problem
and wires domain hooks:
  - simulate(decision) -> traces
  - constraints(traces, decision) -> list[ConstraintResult]
  - objective(traces, constraints, decision) -> ObjectiveResult

This is intentionally a "toy but coherent" microgrid model:
- PV and load are exogenous time series from YAML.
- Decision is battery power schedule (kW), positive = discharge, negative = charge.
- Battery SOC evolves with efficiency and dt.
- Net grid import supplies any remaining load (no export by default; clamp to 0).
- Objective minimizes energy cost + emissions + (optional) unserved energy penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, cast

from gaiaoptics.core.problem import Problem
from gaiaoptics.core.types import (
    ConstraintResult,
    ObjectiveResult,
    Severity,
    TimeIndex,
    clamp,
)


def _get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    """
    Simple nested getter with dot paths (e.g., "battery.capacity_kwh").
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _require_series(name: str, v: Any, n_steps: int) -> List[float]:
    """
    Accept scalar or list[n_steps]. Returns list[n_steps].
    """
    if isinstance(v, (int, float)):
        return [float(v)] * n_steps
    if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
        raise ValueError(f"microgrid cfg requires '{name}' as a number or list[n_steps] of numbers")
    if len(v) != n_steps:
        raise ValueError(f"{name} must have length horizon.n_steps ({n_steps}), got {len(v)}")
    return [float(x) for x in v]


def build_problem_from_config(cfg: Dict[str, Any]) -> Problem:
    """
    Expected minimal YAML shape:

    domain: microgrid
    name: microgrid_dispatch_demo
    time: 0
    horizon:
      n_steps: 24
      dt_hours: 1
    series:
      load_kw: [ ... length n_steps ... ]
      pv_kw:   [ ... length n_steps ... ]
      price_per_kwh: [ ... length n_steps ... ]   # optional; default 0.2
      carbon_kg_per_kwh: [ ... length n_steps ... ] # optional; default 0.4
    battery: ...
    grid: ...
    penalties: ...
    options: ...
    """
    cfg = cfg or {}

    # --- Core identifiers (must exist even for empty cfg) ---
    name = str(cfg.get("name", "microgrid"))

    # --- Horizon ---
    horizon = cfg.get("horizon", {}) or {}
    n_steps = int(horizon.get("n_steps", 24))
    dt_hours = float(horizon.get("dt_hours", 1.0))
    if n_steps <= 0:
        raise ValueError("horizon.n_steps must be > 0")
    if dt_hours <= 0:
        raise ValueError("horizon.dt_hours must be > 0")

    time = TimeIndex(n_steps=n_steps, dt_hours=dt_hours)

    # --- Series (defaults if missing, so empty cfg works) ---
    series = cfg.get("series", {}) or {}

    def _default_series(v: float) -> List[float]:
        return [float(v)] * n_steps

    load_kw = series.get("load_kw", _default_series(10.0))
    pv_kw = series.get("pv_kw", _default_series(0.0))

    price = series.get("price_per_kwh", _default_series(0.2))
    carbon = series.get("carbon_kg_per_kwh", _default_series(0.4))

    load_kw = _require_series("series.load_kw", load_kw, n_steps)
    pv_kw = _require_series("series.pv_kw", pv_kw, n_steps)
    price_per_kwh = _require_series("series.price_per_kwh", price, n_steps)
    carbon_kg_per_kwh = _require_series("series.carbon_kg_per_kwh", carbon, n_steps)

    if len(load_kw) != n_steps:
        raise ValueError("series.load_kw must match horizon.n_steps")
    if len(pv_kw) != n_steps:
        raise ValueError("series.pv_kw must match horizon.n_steps")
    if len(price_per_kwh) != n_steps:
        raise ValueError("series.price_per_kwh must match horizon.n_steps")
    if len(carbon_kg_per_kwh) != n_steps:
        raise ValueError("series.carbon_kg_per_kwh must match horizon.n_steps")

    # --- Battery ---
    battery = cfg.get("battery", {}) or {}
    cap_kwh = float(battery.get("capacity_kwh", 50.0))
    soc0_kwh = float(battery.get("soc0_kwh", cap_kwh / 2.0))
    p_max_kw = float(battery.get("p_max_kw", 25.0))
    eta_c = float(battery.get("eta_charge", 0.95))
    eta_d = float(battery.get("eta_discharge", 0.95))
    soc_min_kwh = float(battery.get("soc_min_kwh", 0.0))
    soc_max_kwh = float(battery.get("soc_max_kwh", cap_kwh))

    # --- Grid / options ---
    grid = cfg.get("grid", {}) or {}
    allow_export = bool(grid.get("allow_export", False))

    options = cfg.get("options", {}) or {}
    allow_unserved = bool(options.get("allow_unserved", False))

    penalties = cfg.get("penalties", {}) or {}
    unserved_penalty_per_kwh = float(penalties.get("unserved_energy_per_kwh", 1000.0))

    # ----------------------------
    # Domain hooks
    # ----------------------------

    def sample_decision(seed: int) -> Dict[str, Any]:
        return {"battery_power_kw": [0.0] * n_steps}

    def repair_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
        p = decision.get("battery_power_kw", None)
        if p is None:
            p = [0.0] * n_steps
        if not isinstance(p, list):
            raise ValueError("decision['battery_power_kw'] must be a list")
        if len(p) != n_steps:
            raise ValueError(f"battery_power_kw must have length {n_steps}, got {len(p)}")
        p2 = [clamp(float(x), -p_max_kw, p_max_kw) for x in p]
        return {**decision, "battery_power_kw": p2}

    def simulate(decision: Dict[str, Any]) -> Dict[str, Any]:
        p_batt = decision["battery_power_kw"]

        soc = float(soc0_kwh)
        soc_series: List[float] = []
        grid_import: List[float] = []
        net_load: List[float] = []
        unserved_kw: List[float] = []
        batt_kw_actual: List[float] = []

        for t in range(n_steps):
            load = float(load_kw[t])
            pv = float(pv_kw[t])
            p = float(p_batt[t])

            batt_kw_actual.append(p)

            net = load - pv - p
            net_load.append(net)

            # toy model: if allow_unserved, we still have no grid cap, so unserved=0
            unserved = 0.0
            unserved_kw.append(unserved)

            if allow_export:
                imp = net
            else:
                imp = max(0.0, net)
            grid_import.append(imp)

            if p >= 0.0:
                soc -= (p / max(eta_d, 1e-9)) * dt_hours
            else:
                soc += (-p * eta_c) * dt_hours
            soc_series.append(soc)

        return {
            "t": list(range(n_steps)),
            "load_kw": load_kw,
            "pv_kw": pv_kw,
            "battery_power_kw": batt_kw_actual,
            "soc_kwh": soc_series,
            "grid_import_kw": grid_import,
            "net_kw": net_load,
            "unserved_kw": unserved_kw,
            "price_per_kwh": price_per_kwh,
            "carbon_kg_per_kwh": carbon_kg_per_kwh,
            "dt_hours": dt_hours,
        }

    def constraints(traces: Dict[str, Any], decision: Dict[str, Any]) -> Sequence[ConstraintResult]:
        soc_series = traces.get("soc_kwh", [])
        if not isinstance(soc_series, list) or len(soc_series) != n_steps:
            raise ValueError("simulate() must return 'soc_kwh' series with length n_steps")

        min_soc_margin = min((float(s) - soc_min_kwh) for s in soc_series)
        max_soc_margin = min((soc_max_kwh - float(s)) for s in soc_series)

        cons: List[ConstraintResult] = [
            ConstraintResult(
                name="battery_soc_min",
                severity=Severity.HARD,
                margin=float(min_soc_margin),
                details={"soc_min_kwh": soc_min_kwh},
            ),
            ConstraintResult(
                name="battery_soc_max",
                severity=Severity.HARD,
                margin=float(max_soc_margin),
                details={"soc_max_kwh": soc_max_kwh},
            ),
        ]

        p = decision["battery_power_kw"]
        p_margin = min(p_max_kw - abs(float(x)) for x in p)
        cons.append(
            ConstraintResult(
                name="battery_power_bounds",
                severity=Severity.HARD,
                margin=float(p_margin),
                details={"p_max_kw": p_max_kw},
            )
        )
        return cons

    def objective(traces: Dict[str, Any], cons: Sequence[ConstraintResult], decision: Dict[str, Any]) -> ObjectiveResult:
        grid_import_kw = traces["grid_import_kw"]
        dt = float(traces["dt_hours"])

        cost = 0.0
        emissions = 0.0
        for t in range(n_steps):
            e_kwh = float(grid_import_kw[t]) * dt
            cost += e_kwh * float(price_per_kwh[t])
            emissions += e_kwh * float(carbon_kg_per_kwh[t])

        soft_pen = 0.0
        for c in cons:
            if c.severity == Severity.SOFT and c.margin < 0.0:
                soft_pen += (-c.margin)

        unserved_kwh = 0.0
        if allow_unserved:
            unserved_series = traces.get("unserved_kw", [0.0] * n_steps)
            if isinstance(unserved_series, list) and len(unserved_series) == n_steps:
                unserved_kwh = sum(float(x) * dt for x in unserved_series)

        unserved_pen = unserved_kwh * unserved_penalty_per_kwh
        score = cost + emissions + (1e3 * soft_pen) + unserved_pen

        return ObjectiveResult(
            score=float(score),
            components={
                "energy_cost": float(cost),
                "emissions_kg": float(emissions),
                "soft_penalty": float(soft_pen),
                "unserved_kwh": float(unserved_kwh),
                "unserved_penalty": float(unserved_pen),
            },
            metadata={
                "score_convention": "lower_is_better",
                "weights": {"cost": 1.0, "emissions": 1.0, "soft_penalty": 1000.0},
            },
        )

    return Problem(
        name=name,
        time=time,
        simulate_fn=simulate,
        constraints_fn=constraints,
        objective_fn=objective,
        sample_decision_fn=sample_decision,
        repair_decision_fn=repair_decision,
        metadata={"domain": "microgrid"},
    )
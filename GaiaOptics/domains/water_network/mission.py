# gaiaoptics/domains/water_network/mission.py
from __future__ import annotations

"""
Water network domain entrypoint (toy but credible).

Minimal tank + pump model with time-varying demand, price, and carbon intensity:
State:
  - tank_level_m3
Decision:
  - pump_power_kw[t] (electric power to pump) in [0, p_max_kw]
Exogenous series:
  - demand_m3ph[t] (water demand outflow)  (default constant)
  - price_per_kwh[t] (default constant)
  - carbon_kg_per_kwh[t] (default constant)

Dynamics (per timestep):
  flow_in_m3ph = pump_power_kw * pump_flow_m3ph_per_kw
  tank_next = tank + (flow_in - demand) * dt_hours

Constraints (HARD):
  - tank_level_min: tank_level_m3[t] >= tank_min_m3
  - tank_level_max: tank_level_m3[t] <= tank_max_m3
  - pump_power_cap: pump_power_kw[t] <= p_max_kw

Objective:
  score = energy_cost + emissions + soft_penalty (future-proof)
  where energy_cost = sum(pump_power_kw[t] * dt_hours * price[t])
        emissions   = sum(pump_power_kw[t] * dt_hours * carbon[t])

Outputs:
  simulate() returns traces series consistent with report/traces.csv usage.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, cast

from gaiaoptics.core.problem import Problem
from gaiaoptics.core.types import ConstraintResult, ObjectiveResult, Severity, TimeIndex, clamp


def _require_series(name: str, v: Any, n_steps: int) -> List[float]:
    """
    Accept scalar or list[n_steps]. Returns list[n_steps].
    """
    if isinstance(v, (int, float)):
        return [float(v)] * n_steps
    if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
        raise ValueError(f"water_network cfg requires '{name}' as a number or list[n_steps] of numbers")
    if len(v) != n_steps:
        raise ValueError(f"{name} must have length horizon.n_steps ({n_steps}), got {len(v)}")
    return [float(x) for x in v]


@dataclass(frozen=True)
class TankParams:
    tank0_m3: float = 50.0
    tank_min_m3: float = 0.0
    tank_max_m3: float = 100.0


@dataclass(frozen=True)
class PumpParams:
    p_max_kw: float = 20.0
    pump_flow_m3ph_per_kw: float = 1.0  # m3/hour per kW


def build_problem_from_config(cfg: Dict[str, Any]) -> Problem:
    cfg = cfg or {}

    name = str(cfg.get("name", "water_network"))

    # Support BOTH styles:
    # - legacy stub: top-level n_steps/dt_hours
    # - canonical: horizon.n_steps/horizon.dt_hours
    horizon = cfg.get("horizon", {}) or {}
    n_steps = int(horizon.get("n_steps", cfg.get("n_steps", 24)))
    dt_hours = float(horizon.get("dt_hours", cfg.get("dt_hours", 1.0)))
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")

    time = TimeIndex(n_steps=n_steps, dt_hours=dt_hours)

    # --- Series ---
    series = cfg.get("series", {}) or {}
    demand_m3ph = _require_series("series.demand_m3ph", series.get("demand_m3ph", 10.0), n_steps)
    price_per_kwh = _require_series("series.price_per_kwh", series.get("price_per_kwh", 0.2), n_steps)
    carbon_kg_per_kwh = _require_series("series.carbon_kg_per_kwh", series.get("carbon_kg_per_kwh", 0.4), n_steps)

    # --- Tank / pump params ---
    tank_cfg = cfg.get("tank", {}) or {}
    tank = TankParams(
        tank0_m3=float(tank_cfg.get("tank0_m3", 50.0)),
        tank_min_m3=float(tank_cfg.get("tank_min_m3", 0.0)),
        tank_max_m3=float(tank_cfg.get("tank_max_m3", 100.0)),
    )
    if tank.tank_max_m3 < tank.tank_min_m3:
        raise ValueError("tank.tank_max_m3 must be >= tank.tank_min_m3")

    pump_cfg = cfg.get("pump", {}) or {}
    pump = PumpParams(
        p_max_kw=float(pump_cfg.get("p_max_kw", 20.0)),
        pump_flow_m3ph_per_kw=float(pump_cfg.get("pump_flow_m3ph_per_kw", 1.0)),
    )
    if pump.p_max_kw <= 0:
        raise ValueError("pump.p_max_kw must be > 0")
    if pump.pump_flow_m3ph_per_kw <= 0:
        raise ValueError("pump.pump_flow_m3ph_per_kw must be > 0")

    def sample_decision(seed: int) -> Dict[str, Any]:
        # Baseline = pump off
        return {"pump_power_kw": [0.0] * time.n_steps}

    def repair_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
        decision = decision or {}
        p = decision.get("pump_power_kw", None)
        if p is None:
            p = [0.0] * time.n_steps
        if not isinstance(p, list):
            raise ValueError("decision['pump_power_kw'] must be a list")
        if len(p) != time.n_steps:
            raise ValueError(f"pump_power_kw must have length {time.n_steps}, got {len(p)}")
        p2 = [clamp(float(x), 0.0, pump.p_max_kw) for x in p]
        return {**decision, "pump_power_kw": p2}

    def simulate(decision: Dict[str, Any]) -> Dict[str, Any]:
        pump_power_kw = decision.get("pump_power_kw")
        if pump_power_kw is None:
            # allow stub-style decisions
            pump_power_kw = [0.0] * time.n_steps
        if not isinstance(pump_power_kw, list) or len(pump_power_kw) != time.n_steps:
            raise ValueError("decision['pump_power_kw'] must be list[n_steps]")

        tank_level = float(tank.tank0_m3)

        tank_level_m3: List[float] = []
        flow_in_m3ph: List[float] = []
        flow_out_m3ph: List[float] = []
        pump_power_series: List[float] = []
        grid_import_kw: List[float] = []

        for t in range(time.n_steps):
            p_kw = float(pump_power_kw[t])
            inflow = p_kw * float(pump.pump_flow_m3ph_per_kw)
            outflow = float(demand_m3ph[t])

            tank_level = tank_level + (inflow - outflow) * float(time.dt_hours)

            tank_level_m3.append(float(tank_level))
            flow_in_m3ph.append(float(inflow))
            flow_out_m3ph.append(float(outflow))
            pump_power_series.append(float(p_kw))
            grid_import_kw.append(float(p_kw))  # electric load

        return {
            "t": list(range(time.n_steps)),
            "dt_hours": float(time.dt_hours),
            "tank_level_m3": tank_level_m3,
            "flow_in_m3ph": flow_in_m3ph,
            "flow_out_m3ph": flow_out_m3ph,
            "pump_power_kw": pump_power_series,
            "grid_import_kw": grid_import_kw,
            "demand_m3ph": [float(x) for x in demand_m3ph],
            "price_per_kwh": [float(x) for x in price_per_kwh],
            "carbon_kg_per_kwh": [float(x) for x in carbon_kg_per_kwh],
        }

    def constraints(traces: Dict[str, Any], decision: Dict[str, Any]) -> Sequence[ConstraintResult]:
        levels = traces.get("tank_level_m3")
        if not isinstance(levels, list) or len(levels) != time.n_steps:
            # preserve old stub behavior if simulate is stubbed elsewhere
            return [ConstraintResult(name="stub_ok", severity=Severity.HARD, margin=1.0, details={})]

        min_margin = min(float(x) - float(tank.tank_min_m3) for x in levels)
        max_margin = min(float(tank.tank_max_m3) - float(x) for x in levels)

        p = traces.get("pump_power_kw", [])
        if isinstance(p, list) and len(p) == time.n_steps:
            p_margin = min(float(pump.p_max_kw) - float(x) for x in p)
        else:
            p_margin = 1.0

        return [
            ConstraintResult(
                name="tank_level_min",
                severity=Severity.HARD,
                margin=float(min_margin),
                details={"tank_min_m3": float(tank.tank_min_m3)},
            ),
            ConstraintResult(
                name="tank_level_max",
                severity=Severity.HARD,
                margin=float(max_margin),
                details={"tank_max_m3": float(tank.tank_max_m3)},
            ),
            ConstraintResult(
                name="pump_power_cap",
                severity=Severity.HARD,
                margin=float(p_margin),
                details={"p_max_kw": float(pump.p_max_kw)},
            ),
        ]

    def objective(traces: Dict[str, Any], cons: Sequence[ConstraintResult], decision: Dict[str, Any]) -> ObjectiveResult:
        # If still stub traces, preserve prior behavior
        if "grid_import_kw" not in traces or "dt_hours" not in traces:
            return ObjectiveResult(
                score=0.0,
                components={"stub": 0.0},
                metadata={"score_convention": "lower_is_better"},
            )

        grid_import_kw = traces["grid_import_kw"]
        dt = float(traces["dt_hours"])

        cost = 0.0
        emissions = 0.0
        for t in range(time.n_steps):
            e_kwh = float(grid_import_kw[t]) * dt
            cost += e_kwh * float(price_per_kwh[t])
            emissions += e_kwh * float(carbon_kg_per_kwh[t])

        # Generic SOFT constraint penalty support
        soft_pen = 0.0
        for c in cons:
            if c.severity == Severity.SOFT and c.margin < 0.0:
                soft_pen += (-c.margin)

        score = cost + emissions + (1e3 * soft_pen)

        return ObjectiveResult(
            score=float(score),
            components={
                "energy_cost": float(cost),
                "emissions_kg": float(emissions),
                "soft_penalty": float(soft_pen),
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
        metadata={"domain": "water_network"},
    )
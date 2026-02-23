# gaiaoptics/domains/data_center/mission.py
from __future__ import annotations

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
        raise ValueError(f"data_center cfg requires '{name}' as a number or list[n_steps] of numbers")
    if len(v) != n_steps:
        raise ValueError(f"{name} must have length horizon.n_steps ({n_steps}), got {len(v)}")
    return [float(x) for x in v]


@dataclass(frozen=True)
class ThermalParams:
    thermal_mass_kwh_per_c: float = 50.0
    ua_kw_per_c: float = 2.0
    cop: float = 3.0


def build_problem_from_config(cfg: Dict[str, Any]) -> Problem:
    cfg = cfg or {}

    name = str(cfg.get("name", "data_center"))

    # Support BOTH styles:
    # - current stub: top-level n_steps/dt_hours
    # - canonical: horizon.n_steps/horizon.dt_hours
    horizon = cfg.get("horizon", {}) or {}
    n_steps = int(horizon.get("n_steps", cfg.get("n_steps", 24)))
    dt_hours = float(horizon.get("dt_hours", cfg.get("dt_hours", 1.0)))
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")

    # Keep TimeIndex type consistent with microgrid
    time = TimeIndex(n_steps=n_steps, dt_hours=dt_hours)

    # --- Series (exogenous) ---
    series = cfg.get("series", {}) or {}
    it_load_kw = _require_series("series.it_load_kw", series.get("it_load_kw", 50.0), n_steps)
    ambient_temp_c = _require_series("series.ambient_temp_c", series.get("ambient_temp_c", 25.0), n_steps)
    price_per_kwh = _require_series("series.price_per_kwh", series.get("price_per_kwh", 0.2), n_steps)
    carbon_kg_per_kwh = _require_series("series.carbon_kg_per_kwh", series.get("carbon_kg_per_kwh", 0.4), n_steps)

    # --- Thermal / cooling ---
    thermal_cfg = cfg.get("thermal", {}) or {}
    temp0_c = float(thermal_cfg.get("temp0_c", 24.0))
    temp_max_c = float(thermal_cfg.get("temp_max_c", 28.0))

    thermal = ThermalParams(
        thermal_mass_kwh_per_c=float(thermal_cfg.get("thermal_mass_kwh_per_c", 50.0)),
        ua_kw_per_c=float(thermal_cfg.get("ua_kw_per_c", 2.0)),
        cop=float(thermal_cfg.get("cop", 3.0)),
    )
    if thermal.thermal_mass_kwh_per_c <= 0:
        raise ValueError("thermal.thermal_mass_kwh_per_c must be > 0")
    if thermal.ua_kw_per_c < 0:
        raise ValueError("thermal.ua_kw_per_c must be >= 0")
    if thermal.cop <= 0:
        raise ValueError("thermal.cop must be > 0")

    cooling_cfg = cfg.get("cooling", {}) or {}
    p_max_kw = float(cooling_cfg.get("p_max_kw", 30.0))
    if p_max_kw <= 0:
        raise ValueError("cooling.p_max_kw must be > 0")

    def sample_decision(seed: int) -> Dict[str, Any]:
        # Baseline default: no cooling
        return {"cooling_power_kw": [0.0] * time.n_steps}

    def repair_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
        decision = decision or {}
        p = decision.get("cooling_power_kw", None)
        if p is None:
            p = [0.0] * time.n_steps
        if not isinstance(p, list):
            raise ValueError("decision['cooling_power_kw'] must be a list")
        if len(p) != time.n_steps:
            raise ValueError(f"cooling_power_kw must have length {time.n_steps}, got {len(p)}")
        p2 = [clamp(float(x), 0.0, p_max_kw) for x in p]
        return {**decision, "cooling_power_kw": p2}

    def simulate(decision: Dict[str, Any]) -> Dict[str, Any]:
        cooling_power_kw = decision.get("cooling_power_kw")
        if cooling_power_kw is None:
            # allow stub-style decisions with empty dict
            cooling_power_kw = [0.0] * time.n_steps
        if not isinstance(cooling_power_kw, list) or len(cooling_power_kw) != time.n_steps:
            raise ValueError("decision['cooling_power_kw'] must be list[n_steps]")

        room_temp_c: List[float] = []
        heat_in_kw_series: List[float] = []
        cooling_removed_kw_series: List[float] = []
        grid_import_kw: List[float] = []

        temp = float(temp0_c)

        for t in range(time.n_steps):
            amb = float(ambient_temp_c[t])
            it_kw = float(it_load_kw[t])
            p_cool = float(cooling_power_kw[t])

            # Envelope heat gain only when ambient > room (stable toy)
            envelope_kw = thermal.ua_kw_per_c * max(0.0, amb - temp)

            heat_in_kw = it_kw + envelope_kw
            cooling_removed_kw = p_cool * thermal.cop

            dtemp = (heat_in_kw - cooling_removed_kw) * float(time.dt_hours) / thermal.thermal_mass_kwh_per_c
            temp = temp + dtemp

            room_temp_c.append(temp)
            heat_in_kw_series.append(heat_in_kw)
            cooling_removed_kw_series.append(cooling_removed_kw)

            # Electricity draw = cooling electric power (toy)
            grid_import_kw.append(p_cool)

        # Minimal traces keyed like microgrid, plus domain-specific series
        return {
            "t": list(range(time.n_steps)),
            "dt_hours": float(time.dt_hours),
            "it_load_kw": [float(x) for x in it_load_kw],
            "ambient_temp_c": [float(x) for x in ambient_temp_c],
            "cooling_power_kw": [float(x) for x in cooling_power_kw],
            "heat_in_kw": heat_in_kw_series,
            "cooling_removed_kw": cooling_removed_kw_series,
            "room_temp_c": room_temp_c,
            "grid_import_kw": grid_import_kw,
            "price_per_kwh": [float(x) for x in price_per_kwh],
            "carbon_kg_per_kwh": [float(x) for x in carbon_kg_per_kwh],
        }

    def constraints(traces: Dict[str, Any], decision: Dict[str, Any]) -> Sequence[ConstraintResult]:
        temps = traces.get("room_temp_c")
        if not isinstance(temps, list) or len(temps) != time.n_steps:
            # Preserve old stub behavior if simulate is still stubbed elsewhere
            return [
                ConstraintResult(
                    name="stub_ok",
                    severity=Severity.HARD,
                    margin=1.0,
                    details={},
                )
            ]

        # margin = min_t (temp_max - temp[t]); negative => violation
        temp_margin = min(float(temp_max_c) - float(x) for x in temps)

        # helpful debugging constraint
        p = traces.get("cooling_power_kw", [])
        if isinstance(p, list) and len(p) == time.n_steps:
            p_margin = min(p_max_kw - float(x) for x in p)
        else:
            p_margin = 1.0

        return [
            ConstraintResult(
                name="temp_max",
                severity=Severity.HARD,
                margin=float(temp_margin),
                details={"temp_max_c": float(temp_max_c)},
            ),
            ConstraintResult(
                name="cooling_power_cap",
                severity=Severity.HARD,
                margin=float(p_margin),
                details={"p_max_kw": float(p_max_kw)},
            ),
        ]

    def objective(
        traces: Dict[str, Any],
        cons: Sequence[ConstraintResult],
        decision: Dict[str, Any],
    ) -> ObjectiveResult:
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

        # Generic soft constraint penalty support
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

    # Some domains may not yet support optional hooks/metadata in Problem signature.
    # Try full signature first, then fall back to minimal.
    try:
        return Problem(
            name=name,
            time=time,
            simulate_fn=simulate,
            constraints_fn=constraints,
            objective_fn=objective,
            sample_decision_fn=sample_decision,
            repair_decision_fn=repair_decision,
            metadata={"domain": "data_center"},
        )
    except TypeError:
        return Problem(
            name=name,
            time=time,
            simulate_fn=simulate,
            constraints_fn=constraints,
            objective_fn=objective,
        )
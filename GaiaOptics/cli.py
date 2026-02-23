# gaiaoptics/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
from random import seed
from typing import Any, Dict, List, Sequence, Tuple

from gaiaoptics.core.config import load_yaml, normalize_config, validate_config
from gaiaoptics.core.errors import ConfigError
from gaiaoptics.core.output import RunArtifacts, write_output_contract
from gaiaoptics.core.types import ConstraintResult, ObjectiveResult, Severity


def _feasibility_from_constraints(cons: Sequence[ConstraintResult]) -> Tuple[bool, str | None, float | None]:
    hard = [c for c in cons if c.severity == Severity.HARD]
    if not hard:
        return True, None, None
    worst = min(hard, key=lambda c: float(getattr(c, "margin", 0.0)))
    worst_margin = float(getattr(worst, "margin", 0.0))
    return worst_margin >= 0.0, str(getattr(worst, "name", "constraint")), worst_margin


def _objective_totals(obj: ObjectiveResult) -> Tuple[float, float]:
    # Prefer explicit components, else fall back to 0.
    comps = obj.components or {}
    cost = float(comps.get("energy_cost", 0.0))
    emissions = float(comps.get("emissions_kg", 0.0))
    return cost, emissions

def _objective_metrics(obj: ObjectiveResult) -> Dict[str, Any]:
    """
    Warehouse Fleet stores metrics under ObjectiveResult.<metrics|details|meta>
    depending on the core schema. This helper safely extracts it.
    """
    for attr in ("metrics", "details", "meta", "components"):
        v = getattr(obj, attr, None)
        if isinstance(v, dict):
            return v
    return {}


def run_microgrid(normalized_cfg: Dict[str, Any]) -> RunArtifacts:
    """
    Engineering-done microgrid runner (Problem-driven, deterministic).

    Works with both config shapes:
      A) domain payload nested: normalized_cfg["microgrid"] has horizon/series/battery/...
      B) domain payload at top-level (legacy): horizon/series/battery at normalized_cfg root
    """
    from gaiaoptics.domains.microgrid.mission import build_problem_from_config

    scenario = normalized_cfg["scenario"]
    planner = normalized_cfg.get("planner", {}) or {}

    # ----------------------------
    # Build a domain payload that matches microgrid/mission.py expectations
    # ----------------------------
    mg_cfg = normalized_cfg.get("microgrid")
    if isinstance(mg_cfg, dict) and ("horizon" in mg_cfg or "series" in mg_cfg or "battery" in mg_cfg):
        payload: Dict[str, Any] = dict(mg_cfg)  # already domain-shaped
    else:
        # Legacy: domain payload at top-level. normalize_config should preserve these keys.
        payload = {"name": str(scenario.get("name", "microgrid"))}
        for k in ("horizon", "series", "battery", "grid", "penalties", "options"):
            v = normalized_cfg.get(k)
            if isinstance(v, (dict, list)):
                payload[k] = v

    missing = [k for k in ("horizon", "series", "battery") if k not in payload]
    if missing:
        raise ConfigError(
            "microgrid",
            f"microgrid payload missing required keys: {missing}",
            hint="Ensure normalize_config() places horizon/series/battery under top-level or under 'microgrid:'",
        )

    problem = build_problem_from_config(payload)
    n = int(problem.time.n_steps)
    dt = float(problem.time.dt_hours)

    # ----------------------------
    # Baseline: sample/repair -> simulate/constraints/objective
    # ----------------------------
    seed = int(scenario.get("seed", 0))
    base_dec = problem.sample_decision_fn(seed) if problem.sample_decision_fn else {}
    base_dec = problem.repair_decision_fn(base_dec) if problem.repair_decision_fn else (base_dec or {})

    base_tr = problem.simulate_fn(base_dec)
    base_cons = problem.constraints_fn(base_tr, base_dec)
    base_obj = problem.objective_fn(base_tr, base_cons, base_dec)

    base_feas, _, _ = _feasibility_from_constraints(base_cons)
    base_cost, base_em = _objective_totals(base_obj)

    # ----------------------------
    # Improved: deterministic heuristic using price median
    # - discharge when price > median
    # - charge when price <= median
    # Try decreasing amplitude until feasible (SOC bounds)
    # ----------------------------
    price = base_tr.get("price_per_kwh")
    if not isinstance(price, list) or len(price) != n:
        price = [0.2] * n

    probe = (
        problem.repair_decision_fn({"battery_power_kw": [1e9] * n})
        if problem.repair_decision_fn
        else {"battery_power_kw": [1e9] * n}
    )
    p_max = float(abs(probe["battery_power_kw"][0]))

    sorted_p = sorted(float(x) for x in price)
    median = sorted_p[len(sorted_p) // 2]

    def schedule(amplitude: float) -> List[float]:
        out: List[float] = []
        for t in range(n):
            out.append(+amplitude if float(price[t]) > median else -amplitude)
        return out

    imp_dec: Dict[str, Any] | None = None
    imp_tr: Dict[str, Any] | None = None
    imp_cons: Sequence[ConstraintResult] = []
    imp_obj: ObjectiveResult | None = None
    imp_feas = False
    imp_worst_name: str | None = None
    imp_worst_margin: float | None = None

    for scale in (0.6, 0.3, 0.15):
        cand = {"battery_power_kw": schedule(scale * p_max)}
        if problem.repair_decision_fn:
            cand = problem.repair_decision_fn(cand)

        tr = problem.simulate_fn(cand)
        cons = problem.constraints_fn(tr, cand)
        feas, worst_name, worst_margin = _feasibility_from_constraints(cons)

        imp_dec, imp_tr, imp_cons = cand, tr, cons
        imp_feas, imp_worst_name, imp_worst_margin = feas, worst_name, worst_margin
        imp_obj = problem.objective_fn(tr, cons, cand)

        if feas:
            break

    imp_cost, imp_em = _objective_totals(imp_obj) if imp_obj is not None else (0.0, 0.0)

    delta_cost = base_cost - imp_cost
    delta_em = base_em - imp_em
    cost_pct = (delta_cost / base_cost * 100.0) if base_cost else 0.0
    em_pct = (delta_em / base_em * 100.0) if base_em else 0.0

    # ----------------------------
    # Traces rows (use improved traces)
    # Include per-step cost/emissions for plot credibility
    # ----------------------------
    assert imp_tr is not None
    t_series = imp_tr.get("t", list(range(n)))
    grid = imp_tr.get("grid_import_kw", [0.0] * n)
    carbon = imp_tr.get("carbon_kg_per_kwh", [0.0] * n)
    price2 = imp_tr.get("price_per_kwh", [0.0] * n)

    for key, arr in (("grid_import_kw", grid), ("carbon_kg_per_kwh", carbon), ("price_per_kwh", price2)):
        if not isinstance(arr, list) or len(arr) != n:
            raise ValueError(f"microgrid simulate() missing/invalid '{key}' series (expected list[n_steps])")

    traces_rows: List[Dict[str, Any]] = []
    load_kw = imp_tr.get("load_kw", [0.0] * n)
    pv_kw = imp_tr.get("pv_kw", [0.0] * n)
    batt_kw = imp_tr.get("battery_power_kw", [0.0] * n)
    soc_kwh = imp_tr.get("soc_kwh", [0.0] * n)

    for series_name, series in (
        ("load_kw", load_kw),
        ("pv_kw", pv_kw),
        ("battery_power_kw", batt_kw),
        ("soc_kwh", soc_kwh),
    ):
        if not isinstance(series, list) or len(series) != n:
            raise ValueError(f"microgrid simulate() missing/invalid '{series_name}' series (expected list[n_steps])")

    for i in range(n):
        e_kwh = float(grid[i]) * dt
        traces_rows.append(
            {
                "t": int(t_series[i]) if isinstance(t_series, list) and i < len(t_series) else i,
                "load_kw": float(load_kw[i]),
                "pv_kw": float(pv_kw[i]),
                "battery_power_kw": float(batt_kw[i]),
                "soc_kwh": float(soc_kwh[i]),
                "grid_import_kw": float(grid[i]),
                "cost": e_kwh * float(price2[i]),
                "emissions": e_kwh * float(carbon[i]),
            }
        )

    solution: Dict[str, Any] = {
        "scenario": scenario["name"],
        "domain": scenario.get("domain", "microgrid"),
        "baseline": {"cost": float(base_cost), "emissions": float(base_em), "feasible": bool(base_feas)},
        "improved": {"cost": float(imp_cost), "emissions": float(imp_em), "feasible": bool(imp_feas)},
        "delta": {
            "cost": float(delta_cost),
            "emissions": float(delta_em),
            "cost_percent_reduction": float(cost_pct),
            "emissions_percent_reduction": float(em_pct),
        },
        "feasibility": {
            "feasible": bool(imp_feas),
            "worst_hard_margin": float(imp_worst_margin) if imp_worst_margin is not None else None,
            "worst_hard_constraint": str(imp_worst_name) if imp_worst_name is not None else None,
        },
        "constraints": [
            {
                "name": c.name,
                "severity": str(c.severity.name),
                "worst_margin": float(c.margin),
                "details": (c.details or {}),
            }
            for c in imp_cons
        ],
        "planner": {
            "name": planner.get("name", "baseline_random"),
            "iterations": int(planner.get("iterations", 200)),
            "restarts": int(planner.get("restarts", 1)),
            "stop_on_feasible": bool(planner.get("stop_on_feasible", False)),
        },
        "traces": {
            "path": "traces.csv",
            "n_rows": len(traces_rows),
            "columns": sorted({k for r in traces_rows for k in r.keys()}),
            "preview": traces_rows[:3],
        },
    }

    # Phase 1 compatibility keys
    solution["feasible"] = bool(solution["feasibility"]["feasible"])
    solution["worst_hard_margin"] = float(solution["feasibility"]["worst_hard_margin"] or 0.0)
    solution["total_cost"] = float(solution["improved"]["cost"])
    solution["total_emissions"] = float(solution["improved"]["emissions"])
    solution["baseline_total_cost"] = float(solution["baseline"]["cost"])
    solution["baseline_total_emissions"] = float(solution["baseline"]["emissions"])

    if imp_obj is not None and imp_obj.components and "unserved_kwh" in imp_obj.components:
        solution["unserved_kwh"] = float(imp_obj.components["unserved_kwh"])

    report_md = (
        f"# Microgrid Dispatch Report — {scenario['name']}\n\n"
        "## Summary\n\n"
        f"- Feasible: {'✅' if solution['feasibility']['feasible'] else '❌'}\n\n"
        "## Cost\n"
        f"- Baseline: ${solution['baseline']['cost']:.2f}\n"
        f"- Improved: ${solution['improved']['cost']:.2f}\n"
        f"- Reduction: ${solution['delta']['cost']:.2f} ({solution['delta']['cost_percent_reduction']:.1f}%)\n\n"
        "## Emissions\n"
        f"- Baseline: {solution['baseline']['emissions']:.2f} kgCO2\n"
        f"- Improved: {solution['improved']['emissions']:.2f} kgCO2\n"
        f"- Reduction: {solution['delta']['emissions']:.2f} ({solution['delta']['emissions_percent_reduction']:.1f}%)\n\n"
        "## Constraint Health\n"
        f"- Worst hard constraint: {solution['feasibility']['worst_hard_constraint']}\n"
        f"- Worst hard margin: {solution['feasibility']['worst_hard_margin']:.6g}\n"
    )

    return RunArtifacts(
        normalized_config=normalized_cfg,
        solution=solution,
        report_md=report_md,
        traces_rows=traces_rows,
    )


def run_data_center(normalized_cfg: Dict[str, Any]) -> RunArtifacts:
    """
    Phase 2 runner for data_center:
    - builds Problem from config
    - baseline + improved decisions are simulated, constrained, and scored
    - solution/report/traces derived from the Problem outputs
    """
    from gaiaoptics.domains.data_center.mission import build_problem_from_config

    scenario = normalized_cfg["scenario"]
    planner = normalized_cfg.get("planner", {}) or {}

    dc_cfg = normalized_cfg.get("data_center") or {}
    if not isinstance(dc_cfg, dict):
        dc_cfg = {}

    problem = build_problem_from_config(dc_cfg)

    base_dec = problem.sample_decision_fn(int(scenario.get("seed", 0))) if problem.sample_decision_fn else {}
    base_dec = problem.repair_decision_fn(base_dec) if problem.repair_decision_fn else (base_dec or {})

    base_tr = problem.simulate_fn(base_dec)
    base_cons = problem.constraints_fn(base_tr, base_dec)
    base_obj = problem.objective_fn(base_tr, base_cons, base_dec)

    base_feas, _, _ = _feasibility_from_constraints(base_cons)
    base_cost, base_em = _objective_totals(base_obj)

    n = int(problem.time.n_steps)

    probe = (
        problem.repair_decision_fn({"cooling_power_kw": [1e9] * n})
        if problem.repair_decision_fn
        else {"cooling_power_kw": [1e9] * n}
    )
    p_max = float(probe["cooling_power_kw"][0])

    imp_dec = {"cooling_power_kw": [0.5 * p_max] * n}
    imp_dec = problem.repair_decision_fn(imp_dec) if problem.repair_decision_fn else imp_dec

    imp_tr = problem.simulate_fn(imp_dec)
    imp_cons = problem.constraints_fn(imp_tr, imp_dec)
    imp_obj = problem.objective_fn(imp_tr, imp_cons, imp_dec)
    imp_feas, imp_worst_name, imp_worst_margin = _feasibility_from_constraints(imp_cons)

    if not imp_feas:
        imp_dec = (
            problem.repair_decision_fn({"cooling_power_kw": [p_max] * n})
            if problem.repair_decision_fn
            else {"cooling_power_kw": [p_max] * n}
        )
        imp_tr = problem.simulate_fn(imp_dec)
        imp_cons = problem.constraints_fn(imp_tr, imp_dec)
        imp_obj = problem.objective_fn(imp_tr, imp_cons, imp_dec)
        imp_feas, imp_worst_name, imp_worst_margin = _feasibility_from_constraints(imp_cons)

    imp_cost, imp_em = _objective_totals(imp_obj)

    delta_cost = base_cost - imp_cost
    delta_em = base_em - imp_em
    cost_pct = (delta_cost / base_cost * 100.0) if base_cost else 0.0
    em_pct = (delta_em / base_em * 100.0) if base_em else 0.0

    t_series = imp_tr.get("t", list(range(n))) if imp_tr is not None else list(range(n))
    dt = float(imp_tr.get("dt_hours", float(problem.time.dt_hours))) if imp_tr is not None else float(problem.time.dt_hours)

    price = imp_tr.get("price_per_kwh", [0.0] * n) if imp_tr is not None else [0.0] * n
    carbon = imp_tr.get("carbon_kg_per_kwh", [0.0] * n) if imp_tr is not None else [0.0] * n
    grid = imp_tr.get("grid_import_kw", [0.0] * n) if imp_tr is not None else [0.0] * n
    temps = imp_tr.get("room_temp_c", [None] * n) if imp_tr is not None else [None] * n
    cool = imp_tr.get("cooling_power_kw", [0.0] * n) if imp_tr is not None else [0.0] * n

    traces_rows: List[Dict[str, Any]] = []
    for i in range(n):
        temp_val = temps[i]
        room_temp_c = float(temp_val) if isinstance(temp_val, (int, float)) else None
        e_kwh = float(grid[i]) * dt
        traces_rows.append(
            {
                "t": int(t_series[i]) if isinstance(t_series, list) and i < len(t_series) else i,
                "room_temp_c": room_temp_c,
                "grid_import_kw": float(grid[i]),
                "cooling_power_kw": float(cool[i]),
                "cost": e_kwh * float(price[i]),
                "emissions": e_kwh * float(carbon[i]),
            }
        )

    solution: Dict[str, Any] = {
        "scenario": scenario["name"],
        "domain": scenario.get("domain", "data_center"),
        "baseline": {"cost": float(base_cost), "emissions": float(base_em), "feasible": bool(base_feas)},
        "improved": {"cost": float(imp_cost), "emissions": float(imp_em), "feasible": bool(imp_feas)},
        "delta": {
            "cost": float(delta_cost),
            "emissions": float(delta_em),
            "cost_percent_reduction": float(cost_pct),
            "emissions_percent_reduction": float(em_pct),
        },
        "feasibility": {
            "feasible": bool(imp_feas),
            "worst_hard_margin": float(imp_worst_margin) if imp_worst_margin is not None else None,
            "worst_hard_constraint": str(imp_worst_name) if imp_worst_name is not None else None,
        },
        "constraints": [
            {
                "name": c.name,
                "severity": str(c.severity.name),
                "worst_margin": float(c.margin),
                "details": (c.details or {}),
            }
            for c in imp_cons
        ],
        "planner": {
            "name": planner.get("name", "baseline_random"),
            "iterations": int(planner.get("iterations", 200)),
            "restarts": int(planner.get("restarts", 1)),
            "stop_on_feasible": bool(planner.get("stop_on_feasible", False)),
        },
        "traces": {
            "path": "traces.csv",
            "n_rows": len(traces_rows),
            "columns": sorted({k for r in traces_rows for k in r.keys()}),
            "preview": traces_rows[:3],
        },
    }

    solution["feasible"] = bool(solution["feasibility"]["feasible"])
    solution["worst_hard_margin"] = float(solution["feasibility"]["worst_hard_margin"] or 0.0)
    solution["total_cost"] = float(solution["improved"]["cost"])
    solution["total_emissions"] = float(solution["improved"]["emissions"])
    solution["baseline_total_cost"] = float(solution["baseline"]["cost"])
    solution["baseline_total_emissions"] = float(solution["baseline"]["emissions"])

    report_md = (
        f"# Data Center Thermal Report — {scenario['name']}\n\n"
        "## Summary\n\n"
        f"- Feasible: {'✅' if solution['feasibility']['feasible'] else '❌'}\n\n"
        "## Cost\n"
        f"- Baseline: ${solution['baseline']['cost']:.2f}\n"
        f"- Improved: ${solution['improved']['cost']:.2f}\n"
        f"- Reduction: ${solution['delta']['cost']:.2f} ({solution['delta']['cost_percent_reduction']:.1f}%)\n\n"
        "## Emissions\n"
        f"- Baseline: {solution['baseline']['emissions']:.2f} kgCO2\n"
        f"- Improved: {solution['improved']['emissions']:.2f} kgCO2\n"
        f"- Reduction: {solution['delta']['emissions']:.2f} ({solution['delta']['emissions_percent_reduction']:.1f}%)\n\n"
        "## Constraint Health\n"
        f"- Worst hard constraint: {solution['feasibility']['worst_hard_constraint']}\n"
        f"- Worst hard margin: {solution['feasibility']['worst_hard_margin']:.6g}\n"
    )

    return RunArtifacts(
        normalized_config=normalized_cfg,
        solution=solution,
        report_md=report_md,
        traces_rows=traces_rows,
    )


def run_water_network(normalized_cfg: Dict[str, Any]) -> RunArtifacts:
    """
    Phase 2 runner for water_network:
    - builds Problem from config
    - baseline + improved decisions are simulated, constrained, and scored
    - feasibility + worst constraint derived from HARD margins
    - traces.csv includes tank_level_m3 for plotting
    """
    from gaiaoptics.domains.water_network.mission import build_problem_from_config

    scenario = normalized_cfg["scenario"]
    planner = normalized_cfg.get("planner", {}) or {}

    wn_cfg = normalized_cfg.get("water_network") or {}
    if not isinstance(wn_cfg, dict):
        wn_cfg = {}

    problem = build_problem_from_config(wn_cfg)
    n = int(problem.time.n_steps)

    base_dec = problem.sample_decision_fn(int(scenario.get("seed", 0))) if problem.sample_decision_fn else {}
    base_dec = problem.repair_decision_fn(base_dec) if problem.repair_decision_fn else (base_dec or {})

    base_tr = problem.simulate_fn(base_dec)
    base_cons = problem.constraints_fn(base_tr, base_dec)
    base_obj = problem.objective_fn(base_tr, base_cons, base_dec)

    base_feas, _, _ = _feasibility_from_constraints(base_cons)
    base_cost, base_em = _objective_totals(base_obj)

    probe = (
        problem.repair_decision_fn({"pump_power_kw": [1e9] * n})
        if problem.repair_decision_fn
        else {"pump_power_kw": [1e9] * n}
    )
    flow_probe_dec = (
        problem.repair_decision_fn({"pump_power_kw": [1.0] * n})
        if problem.repair_decision_fn
        else {"pump_power_kw": [1.0] * n}
    )
    flow_probe_tr = problem.simulate_fn(flow_probe_dec)
    flow_in = flow_probe_tr.get("flow_in_m3ph", [1.0] * n)
    flow_per_kw = float(flow_in[0]) if isinstance(flow_in, list) and flow_in else 1.0
    if flow_per_kw <= 0:
        flow_per_kw = 1.0

    demand = base_tr.get("demand_m3ph", [10.0] * n)
    if not isinstance(demand, list) or len(demand) != n:
        demand = [10.0] * n

    pump_kw_series = []
    for t in range(n):
        target_kw = float(demand[t]) / flow_per_kw
        target_kw *= 1.05
        pump_kw_series.append(target_kw)

    imp_dec = (
        problem.repair_decision_fn({"pump_power_kw": pump_kw_series})
        if problem.repair_decision_fn
        else {"pump_power_kw": pump_kw_series}
    )
    imp_tr = problem.simulate_fn(imp_dec)
    imp_cons = problem.constraints_fn(imp_tr, imp_dec)
    imp_obj = problem.objective_fn(imp_tr, imp_cons, imp_dec)

    imp_feas, imp_worst_name, imp_worst_margin = _feasibility_from_constraints(imp_cons)
    imp_cost, imp_em = _objective_totals(imp_obj)

    if not imp_feas:
        pump_kw_series = []
        for t in range(n):
            target_kw = float(demand[t]) / flow_per_kw
            target_kw *= 0.98
            pump_kw_series.append(target_kw)

        imp_dec = (
            problem.repair_decision_fn({"pump_power_kw": pump_kw_series})
            if problem.repair_decision_fn
            else {"pump_power_kw": pump_kw_series}
        )
        imp_tr = problem.simulate_fn(imp_dec)
        imp_cons = problem.constraints_fn(imp_tr, imp_dec)
        imp_obj = problem.objective_fn(imp_tr, imp_cons, imp_dec)

        imp_feas, imp_worst_name, imp_worst_margin = _feasibility_from_constraints(imp_cons)
        imp_cost, imp_em = _objective_totals(imp_obj)

    delta_cost = base_cost - imp_cost
    delta_em = base_em - imp_em
    cost_pct = (delta_cost / base_cost * 100.0) if base_cost else 0.0
    em_pct = (delta_em / base_em * 100.0) if base_em else 0.0

    t_series = imp_tr.get("t", list(range(n))) if imp_tr is not None else list(range(n))
    dt = float(imp_tr.get("dt_hours", float(problem.time.dt_hours))) if imp_tr is not None else float(problem.time.dt_hours)

    price = imp_tr.get("price_per_kwh", [0.0] * n) if imp_tr is not None else [0.0] * n
    carbon = imp_tr.get("carbon_kg_per_kwh", [0.0] * n) if imp_tr is not None else [0.0] * n
    grid = imp_tr.get("grid_import_kw", [0.0] * n) if imp_tr is not None else [0.0] * n
    tank = imp_tr.get("tank_level_m3", [None] * n) if imp_tr is not None else [None] * n
    pump = imp_tr.get("pump_power_kw", [0.0] * n) if imp_tr is not None else [0.0] * n
    dem2 = imp_tr.get("demand_m3ph", [0.0] * n) if imp_tr is not None else [0.0] * n

    traces_rows: List[Dict[str, Any]] = []
    for i in range(n):
        tank_val = tank[i]
        tank_level_m3 = float(tank_val) if isinstance(tank_val, (int, float)) else None
        e_kwh = float(grid[i]) * dt
        traces_rows.append(
            {
                "t": int(t_series[i]) if isinstance(t_series, list) and i < len(t_series) else i,
                "tank_level_m3": tank_level_m3,
                "pump_power_kw": float(pump[i]),
                "demand_m3ph": float(dem2[i]),
                "cost": e_kwh * float(price[i]),
                "emissions": e_kwh * float(carbon[i]),
            }
        )

    solution: Dict[str, Any] = {
        "scenario": scenario["name"],
        "domain": scenario.get("domain", "water_network"),
        "baseline": {"cost": float(base_cost), "emissions": float(base_em), "feasible": bool(base_feas)},
        "improved": {"cost": float(imp_cost), "emissions": float(imp_em), "feasible": bool(imp_feas)},
        "delta": {
            "cost": float(delta_cost),
            "emissions": float(delta_em),
            "cost_percent_reduction": float(cost_pct),
            "emissions_percent_reduction": float(em_pct),
        },
        "feasibility": {
            "feasible": bool(imp_feas),
            "worst_hard_margin": float(imp_worst_margin) if imp_worst_margin is not None else None,
            "worst_hard_constraint": str(imp_worst_name) if imp_worst_name is not None else None,
        },
        "constraints": [
            {
                "name": c.name,
                "severity": str(c.severity.name),
                "worst_margin": float(c.margin),
                "details": (c.details or {}),
            }
            for c in imp_cons
        ],
        "planner": {
            "name": planner.get("name", "baseline_random"),
            "iterations": int(planner.get("iterations", 200)),
            "restarts": int(planner.get("restarts", 1)),
            "stop_on_feasible": bool(planner.get("stop_on_feasible", False)),
        },
        "traces": {
            "path": "traces.csv",
            "n_rows": len(traces_rows),
            "columns": sorted({k for r in traces_rows for k in r.keys()}),
            "preview": traces_rows[:3],
        },
    }

    solution["feasible"] = bool(solution["feasibility"]["feasible"])
    solution["worst_hard_margin"] = float(solution["feasibility"]["worst_hard_margin"] or 0.0)
    solution["total_cost"] = float(solution["improved"]["cost"])
    solution["total_emissions"] = float(solution["improved"]["emissions"])
    solution["baseline_total_cost"] = float(solution["baseline"]["cost"])
    solution["baseline_total_emissions"] = float(solution["baseline"]["emissions"])

    report_md = (
        f"# Water Network Operations Report — {scenario['name']}\n\n"
        "## Summary\n\n"
        f"- Feasible: {'✅' if solution['feasibility']['feasible'] else '❌'}\n\n"
        "## Cost\n"
        f"- Baseline: ${solution['baseline']['cost']:.2f}\n"
        f"- Improved: ${solution['improved']['cost']:.2f}\n"
        f"- Reduction: ${solution['delta']['cost']:.2f} ({solution['delta']['cost_percent_reduction']:.1f}%)\n\n"
        "## Emissions\n"
        f"- Baseline: {solution['baseline']['emissions']:.2f} kgCO2\n"
        f"- Improved: {solution['improved']['emissions']:.2f} kgCO2\n"
        f"- Reduction: ${solution['delta']['emissions']:.2f} ({solution['delta']['emissions_percent_reduction']:.1f}%)\n\n"
        "## Constraint Health\n"
        f"- Worst hard constraint: {solution['feasibility']['worst_hard_constraint']}\n"
        f"- Worst hard margin: {solution['feasibility']['worst_hard_margin']:.6g}\n"
    )

    return RunArtifacts(
        normalized_config=normalized_cfg,
        solution=solution,
        report_md=report_md,
        traces_rows=traces_rows,
    )


def run_warehouse_fleet(normalized_cfg: Dict[str, Any]) -> RunArtifacts:
    """
    Phase 2 runner for warehouse_fleet:
    - builds Problem via domains.warehouse_fleet.mission.build_problem(cfg)
    - baseline + improved decisions are simulated, constrained, and scored
    - decisions must include: {"assignments": list[list[int]]}
    - warehouse_fleet currently reports energy + distance (not $ cost / CO2)
    """
    from gaiaoptics.domains.warehouse_fleet.mission import build_problem as build_warehouse_fleet_problem

    scenario = normalized_cfg["scenario"]
    planner = normalized_cfg.get("planner", {}) or {}

    problem = build_warehouse_fleet_problem(normalized_cfg)
    seed = int(scenario.get("seed", 0))
    n = int(problem.time.n_steps)

    def _as_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _fmt(x: Any, fmt: str, default: str = "—") -> str:
        if x is None:
            return default
        try:
            return format(float(x), fmt)
        except Exception:
            return default

    def _objective_scalar(obj: ObjectiveResult) -> float:
        """
        Compare candidates by scalar objective.
        Prefer .value then .score; else +inf.
        """
        for attr in ("value", "score"):
            v = getattr(obj, attr, None)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str) and v.strip():
                try:
                    return float(v)
                except Exception:
                    pass
        return float("inf")

    def _safe_eval(decision: Dict[str, Any]):
        """
        Returns tuple:
          (decision, traces, constraints, objective, feasible, worst_name, worst_margin)
        If simulate blows up, returns infeasible placeholders.
        """
        try:
            dec = decision
            if problem.repair_decision_fn:
                dec = problem.repair_decision_fn(dec)

            tr = problem.simulate_fn(dec)
            cons = problem.constraints_fn(tr, dec)
            obj = problem.objective_fn(tr, cons, dec)
            feas, worst_name, worst_margin = _feasibility_from_constraints(cons)
            return dec, tr, cons, obj, feas, worst_name, worst_margin

        except Exception:
            empty_tr: Dict[str, Any] = {"t": list(range(n))}
            empty_cons: List[ConstraintResult] = []
            try:
                empty_obj = ObjectiveResult(value=float("inf"), components={})  # type: ignore[arg-type]
            except TypeError:
                empty_obj = ObjectiveResult(score=float("inf"), components={})  # type: ignore[arg-type]
            return decision, empty_tr, empty_cons, empty_obj, False, "exception", None

    def _energy_distance_from_traces(tr: Dict[str, Any]) -> Tuple[float, float]:
        return (
            _as_float(tr.get("energy_used", 0.0), 0.0),
            _as_float(tr.get("distance_total", 0.0), 0.0),
        )

    # ----------------------------
    # Baseline
    # ----------------------------
    if not problem.sample_decision_fn:
        raise ConfigError(
            "warehouse_fleet",
            "warehouse_fleet Problem missing sample_decision_fn",
            hint="Implement tasks.sample_decision(cfg, seed) and wire it in domains/warehouse_fleet/mission.py",
        )

    base_dec0 = problem.sample_decision_fn(seed)
    base_dec, base_tr, base_cons, base_obj, base_feas, base_worst_name, base_worst_margin = _safe_eval(base_dec0)

    base_energy, base_distance = _energy_distance_from_traces(base_tr if isinstance(base_tr, dict) else {})
    base_score = _objective_scalar(base_obj)

    # ----------------------------
    # Improved: try a few deterministic seeds, pick best feasible then lowest score
    # ----------------------------
    best = (base_dec, base_tr, base_cons, base_obj, base_feas, base_worst_name, base_worst_margin)

    iters = int(planner.get("iterations", 200))
    # sample iters candidates deterministically
    for k in range(1, iters + 1):
        s = seed + k
        cand0 = problem.sample_decision_fn(s)
        cand = _safe_eval(cand0)

        _, _, _, cand_obj, cand_feas, _, _ = cand
        _, _, _, best_obj, best_feas, _, _ = best

        if cand_feas and not best_feas:
            best = cand
        elif cand_feas == best_feas and _objective_scalar(cand_obj) < _objective_scalar(best_obj):
            best = cand

    imp_dec, imp_tr, imp_cons, imp_obj, imp_feas, imp_worst_name, imp_worst_margin = best

    imp_energy, imp_distance = _energy_distance_from_traces(imp_tr if isinstance(imp_tr, dict) else {})
    imp_score = _objective_scalar(imp_obj)

    delta_energy = base_energy - imp_energy
    delta_distance = base_distance - imp_distance
    energy_pct = (delta_energy / base_energy * 100.0) if base_energy else 0.0
    dist_pct = (delta_distance / base_distance * 100.0) if base_distance else 0.0

    # ----------------------------
    # Traces rows: minimal stable CSV
    # ----------------------------
    t_series_raw = imp_tr.get("t") if isinstance(imp_tr, dict) else None
    if isinstance(t_series_raw, list) and len(t_series_raw) == n:
        try:
            t_series = [int(float(x)) for x in t_series_raw]
        except Exception:
            t_series = list(range(n))
    else:
        t_series = list(range(n))

    traces_rows: List[Dict[str, Any]] = [{"t": t_series[i]} for i in range(n)]

    solution: Dict[str, Any] = {
        "scenario": scenario["name"],
        "domain": scenario.get("domain", "warehouse_fleet"),

        "baseline": {
            "energy_used": float(base_energy),
            "distance_total": float(base_distance),
            "score": float(base_score),
            "feasible": bool(base_feas),
        },
        "improved": {
            "energy_used": float(imp_energy),
            "distance_total": float(imp_distance),
            "score": float(imp_score),
            "feasible": bool(imp_feas),
        },
        "delta": {
            "energy_used": float(delta_energy),
            "energy_percent_reduction": float(energy_pct),
            "distance_total": float(delta_distance),
            "distance_percent_reduction": float(dist_pct),
        },

        "feasibility": {
            "feasible": bool(imp_feas),
            "worst_hard_margin": float(imp_worst_margin) if imp_worst_margin is not None else None,
            "worst_hard_constraint": str(imp_worst_name) if imp_worst_name is not None else None,
        },

        "constraints": [
            {
                "name": c.name,
                "severity": str(c.severity.name),
                "worst_margin": float(c.margin),
                "details": (c.details or {}),
            }
            for c in imp_cons
        ],

        "planner": {
            "name": planner.get("name", "baseline_random"),
            "iterations": int(planner.get("iterations", 200)),
            "restarts": int(planner.get("restarts", 1)),
            "stop_on_feasible": bool(planner.get("stop_on_feasible", False)),
        },

        "decision": imp_dec,

        "traces": {
            "path": "traces.csv",
            "n_rows": len(traces_rows),
            "columns": sorted({k for r in traces_rows for k in r.keys()}),
            "preview": traces_rows[:3],
        },
    }

    # Phase 1 compatibility keys (truthful mapping)
    solution["feasible"] = bool(solution["feasibility"]["feasible"])
    solution["worst_hard_margin"] = float(solution["feasibility"]["worst_hard_margin"] or 0.0)

    # Microgrid-era names: map "cost" -> energy proxy; emissions not modeled
    solution["total_cost"] = float(solution["improved"]["energy_used"])
    solution["baseline_total_cost"] = float(solution["baseline"]["energy_used"])
    solution["total_emissions"] = 0.0
    solution["baseline_total_emissions"] = 0.0

    report_md = (
        f"# Warehouse Fleet Report — {scenario.get('name', 'scenario')}\n\n"
        "## Summary\n\n"
        f"- Feasible: {'✅' if solution['feasibility']['feasible'] else '❌'}\n"
        f"- Score: {_fmt(solution.get('improved', {}).get('score'), '.6g')}\n\n"
        "## Energy\n"
        f"- Baseline: {_fmt(solution.get('baseline', {}).get('energy_used'), '.3f')}\n"
        f"- Improved: {_fmt(solution.get('improved', {}).get('energy_used'), '.3f')}\n"
        f"- Reduction: {_fmt(solution.get('delta', {}).get('energy_used'), '.3f')} ({_fmt(solution.get('delta', {}).get('energy_percent_reduction'), '.1f')}%)\n\n"
        "## Distance\n"
        f"- Baseline: {_fmt(solution.get('baseline', {}).get('distance_total'), '.3f')}\n"
        f"- Improved: {_fmt(solution.get('improved', {}).get('distance_total'), '.3f')}\n"
        f"- Reduction: {_fmt(solution.get('delta', {}).get('distance_total'), '.3f')} ({_fmt(solution.get('delta', {}).get('distance_percent_reduction'), '.1f')}%)\n\n"
        "## Constraint Health\n"
        f"- Worst hard constraint: {solution.get('feasibility', {}).get('worst_hard_constraint')}\n"
        f"- Worst hard margin: {_fmt(solution.get('feasibility', {}).get('worst_hard_margin'), '.6g')}\n"
    )

    return RunArtifacts(
        normalized_config=normalized_cfg,
        solution=solution,
        report_md=report_md,
        traces_rows=traces_rows,
    )


def _run_domain(normalized_cfg: Dict[str, Any]) -> RunArtifacts:
    dom = normalized_cfg["scenario"].get("domain", "microgrid")
    if dom == "microgrid":
        return run_microgrid(normalized_cfg)
    if dom == "data_center":
        return run_data_center(normalized_cfg)
    if dom == "water_network":
        return run_water_network(normalized_cfg)
    if dom == "warehouse_fleet":
        return run_warehouse_fleet(normalized_cfg)
    raise ConfigError("scenario.domain", f"unsupported domain '{dom}'")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gaiaoptics")
    ap.add_argument("yaml_path", type=str)

    ap.add_argument("--out", type=str, default=None, help="Output root directory (preferred)")
    ap.add_argument("--outputs-root", type=str, default=None, help="Output root directory (alias)")

    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stop-on-feasible", action="store_true")

    args = ap.parse_args(argv)

    outputs_root = args.out or args.outputs_root or "outputs"
    ypath = Path(args.yaml_path)

    try:
        raw = load_yaml(ypath)
        normalized = normalize_config(raw, source_path=ypath)

        # Apply CLI overrides into normalized config
        if args.seed is not None:
            normalized["scenario"]["seed"] = int(args.seed)
        if args.iterations is not None:
            normalized["planner"]["iterations"] = int(args.iterations)
        if args.restarts is not None:
            normalized["planner"]["restarts"] = int(args.restarts)
        if args.stop_on_feasible:
            normalized["planner"]["stop_on_feasible"] = True

        validate_config(normalized)

        scenario_name = normalized["scenario"]["name"]
        artifacts = _run_domain(normalized)

        write_output_contract(Path(outputs_root), scenario_name, artifacts)
        return 0

    except ConfigError as e:
        print(str(e))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
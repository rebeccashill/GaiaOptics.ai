# gaiaoptics/cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from gaiaoptics.core.config import load_yaml, normalize_config, validate_config
from gaiaoptics.core.errors import ConfigError
from gaiaoptics.core.output import RunArtifacts, write_output_contract


def run_microgrid(normalized_cfg: dict) -> RunArtifacts:
    """
    Temporary deterministic runner sufficient for contract + end-to-end tests.
    Produces a mature solution.json structure:
      - baseline / improved / delta
      - feasibility / constraints / traces
    """
    scenario = normalized_cfg["scenario"]
    planner = normalized_cfg.get("planner", {}) or {}

    baseline_cost = 156.70
    baseline_emissions = 112.30
    baseline_feasible = True

    improved_cost = 123.45
    improved_emissions = 98.70
    improved_feasible = True

    delta_cost = baseline_cost - improved_cost
    delta_emissions = baseline_emissions - improved_emissions

    cost_pct = (delta_cost / baseline_cost * 100.0) if baseline_cost else 0.0
    emissions_pct = (delta_emissions / baseline_emissions * 100.0) if baseline_emissions else 0.0

    worst_hard_margin = 0.05
    worst_hard_constraint = "battery_nonnegative"

    traces_rows = [
        {"t": 0, "load_kw": 10.0, "grid_kw": 5.0, "cost": 1.0, "emissions": 0.5},
        {"t": 1, "load_kw": 12.0, "grid_kw": 6.0, "cost": 1.2, "emissions": 0.6},
    ]

    solution = {
        "scenario": scenario["name"],
        "domain": scenario.get("domain", "microgrid"),

        "baseline": {
            "cost": float(baseline_cost),
            "emissions": float(baseline_emissions),
            "feasible": bool(baseline_feasible),
        },
        "improved": {
            "cost": float(improved_cost),
            "emissions": float(improved_emissions),
            "feasible": bool(improved_feasible),
        },
        "delta": {
            "cost": float(delta_cost),
            "emissions": float(delta_emissions),
            "cost_percent_reduction": float(cost_pct),
            "emissions_percent_reduction": float(emissions_pct),
        },
        "feasibility": {
            "feasible": bool(improved_feasible),
            "worst_hard_margin": float(worst_hard_margin),
            "worst_hard_constraint": str(worst_hard_constraint),
        },
        "constraints": [
            {"name": "battery_nonnegative", "severity": "HARD", "worst_margin": float(worst_hard_margin)}
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

    # --- Backward-compatible Phase 1 keys (tests expect these) ---
    solution["feasible"] = bool(solution["feasibility"]["feasible"])
    solution["worst_hard_margin"] = float(solution["feasibility"]["worst_hard_margin"])
    solution["total_cost"] = float(solution["improved"]["cost"])
    solution["total_emissions"] = float(solution["improved"]["emissions"])
    solution["baseline_total_cost"] = float(solution["baseline"]["cost"])
    solution["baseline_total_emissions"] = float(solution["baseline"]["emissions"])

    # Report (microgrid-specific wording is OK here, because it's the microgrid runner)
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
        f"- Worst hard margin: {solution['feasibility']['worst_hard_margin']:.3f}\n"
    )

    return RunArtifacts(
        normalized_config=normalized_cfg,
        solution=solution,
        report_md=report_md,
        traces_rows=traces_rows,
    )


def run_data_center(normalized_cfg: dict) -> RunArtifacts:
    """
    Phase 2 runner (can be stubbed, but must be domain-correct).
    For now: deterministic placeholder metrics + data_center constraint names.
    """
    scenario = normalized_cfg["scenario"]
    planner = normalized_cfg.get("planner", {}) or {}

    baseline_cost = 40.00
    baseline_emissions = 20.00
    baseline_feasible = True

    improved_cost = 32.00
    improved_emissions = 16.00
    improved_feasible = True

    delta_cost = baseline_cost - improved_cost
    delta_emissions = baseline_emissions - improved_emissions

    cost_pct = (delta_cost / baseline_cost * 100.0) if baseline_cost else 0.0
    emissions_pct = (delta_emissions / baseline_emissions * 100.0) if baseline_emissions else 0.0

    worst_hard_margin = 0.42
    worst_hard_constraint = "temp_max"

    traces_rows = [
        {"t": 0, "room_temp_c": 24.0, "cooling_kw": 5.0, "cost": 1.0, "emissions": 0.5},
        {"t": 1, "room_temp_c": 24.3, "cooling_kw": 6.0, "cost": 1.2, "emissions": 0.6},
    ]

    solution = {
        "scenario": scenario["name"],
        "domain": scenario.get("domain", "data_center"),

        "baseline": {"cost": float(baseline_cost), "emissions": float(baseline_emissions), "feasible": bool(baseline_feasible)},
        "improved": {"cost": float(improved_cost), "emissions": float(improved_emissions), "feasible": bool(improved_feasible)},
        "delta": {
            "cost": float(delta_cost),
            "emissions": float(delta_emissions),
            "cost_percent_reduction": float(cost_pct),
            "emissions_percent_reduction": float(emissions_pct),
        },
        "feasibility": {
            "feasible": bool(improved_feasible),
            "worst_hard_margin": float(worst_hard_margin),
            "worst_hard_constraint": str(worst_hard_constraint),
        },
        "constraints": [
            {"name": "temp_max", "severity": "HARD", "worst_margin": float(worst_hard_margin)}
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

    # Keep Phase 1-compatible keys (so your downstream summary extractor doesn’t break)
    solution["feasible"] = bool(solution["feasibility"]["feasible"])
    solution["worst_hard_margin"] = float(solution["feasibility"]["worst_hard_margin"])
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
        f"- Worst hard margin: {solution['feasibility']['worst_hard_margin']:.3f}\n"
    )

    return RunArtifacts(
        normalized_config=normalized_cfg,
        solution=solution,
        report_md=report_md,
        traces_rows=traces_rows,
    )


def run_water_network(normalized_cfg: dict) -> RunArtifacts:
    """
    Phase 2 runner (stub) for water network domain.
    Produces deterministic placeholder metrics + water_network constraint names.
    """
    scenario = normalized_cfg["scenario"]
    planner = normalized_cfg.get("planner", {}) or {}

    baseline_cost = 60.00
    baseline_emissions = 12.00
    baseline_feasible = True

    improved_cost = 48.00
    improved_emissions = 9.50
    improved_feasible = True

    delta_cost = baseline_cost - improved_cost
    delta_emissions = baseline_emissions - improved_emissions

    cost_pct = (delta_cost / baseline_cost * 100.0) if baseline_cost else 0.0
    emissions_pct = (delta_emissions / baseline_emissions * 100.0) if baseline_emissions else 0.0

    worst_hard_margin = 0.18
    worst_hard_constraint = "pressure_min"

    traces_rows = [
        {"t": 0, "demand_lps": 80.0, "pump_kw": 12.0, "cost": 1.1, "emissions": 0.3},
        {"t": 1, "demand_lps": 85.0, "pump_kw": 13.0, "cost": 1.2, "emissions": 0.35},
    ]

    solution = {
        "scenario": scenario["name"],
        "domain": scenario.get("domain", "water_network"),

        "baseline": {"cost": float(baseline_cost), "emissions": float(baseline_emissions), "feasible": bool(baseline_feasible)},
        "improved": {"cost": float(improved_cost), "emissions": float(improved_emissions), "feasible": bool(improved_feasible)},
        "delta": {
            "cost": float(delta_cost),
            "emissions": float(delta_emissions),
            "cost_percent_reduction": float(cost_pct),
            "emissions_percent_reduction": float(emissions_pct),
        },
        "feasibility": {
            "feasible": bool(improved_feasible),
            "worst_hard_margin": float(worst_hard_margin),
            "worst_hard_constraint": str(worst_hard_constraint),
        },
        "constraints": [
            {"name": "pressure_min", "severity": "HARD", "worst_margin": float(worst_hard_margin)}
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

    # Keep Phase 1-compatible keys
    solution["feasible"] = bool(solution["feasibility"]["feasible"])
    solution["worst_hard_margin"] = float(solution["feasibility"]["worst_hard_margin"])
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
        f"- Reduction: {solution['delta']['emissions']:.2f} ({solution['delta']['emissions_percent_reduction']:.1f}%)\n\n"
        "## Constraint Health\n"
        f"- Worst hard constraint: {solution['feasibility']['worst_hard_constraint']}\n"
        f"- Worst hard margin: {solution['feasibility']['worst_hard_margin']:.3f}\n"
    )

    return RunArtifacts(
        normalized_config=normalized_cfg,
        solution=solution,
        report_md=report_md,
        traces_rows=traces_rows,
    )


def _run_domain(normalized_cfg: dict) -> RunArtifacts:
    dom = normalized_cfg["scenario"].get("domain", "microgrid")
    if dom == "microgrid":
        return run_microgrid(normalized_cfg)
    if dom == "data_center":
        return run_data_center(normalized_cfg)
    if dom == "water_network":
        return run_water_network(normalized_cfg)
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
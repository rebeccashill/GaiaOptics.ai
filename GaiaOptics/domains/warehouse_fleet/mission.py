# gaiaoptics/domains/warehouse_fleet/mission.py
"""
Warehouse Fleet domain: Problem builder + hook wiring.

This module is the single entry-point the core/CLI should call to obtain a
domain-agnostic `Problem` with the required hook surface:

  decision -> simulate -> constraints -> objective

Expected sibling modules (you can stub these first, then flesh them out):
  - mobility.py:   simulate(cfg, decision, time) -> traces
  - constraints.py: evaluate(cfg, traces, decision) -> list[ConstraintResult]
  - objective.py:  evaluate(cfg, traces, constraints, decision) -> ObjectiveResult
  - tasks.py:      sample_decision(cfg, seed) -> decision
                  (optional) repair_decision(cfg, decision) -> decision

Decision contract (recommended v1):
  decision = {"assignments": list[list[int]]}  # tasks per robot, in visit order

Traces contract (recommended v1):
  traces = {
    "energy_used": float,
    "tasks_completed": int,
    "robot_battery_min": float,
    "distance_total": float,
    ...
  }
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from gaiaoptics.core.problem import Problem
from gaiaoptics.core.types import TimeIndex

from . import constraints as wf_constraints
from . import mobility as wf_mobility
from . import objective as wf_objective
from . import tasks as wf_tasks


def _make_time_index(wf_cfg: Dict[str, Any]) -> TimeIndex:
    """
    Convert warehouse_fleet config into a core TimeIndex.

    For a discrete grid fleet sim, this is typically 0..T-1 steps.
    Adjust if your TimeIndex type differs.
    """
    horizon_steps = int(wf_cfg.get("horizon_steps", 200))
    if horizon_steps <= 0:
        raise ValueError("warehouse_fleet.horizon_steps must be > 0")
    # TimeIndex in your project appears to be a type imported from core.types.
    # If it's a dataclass/alias around a list, this will work.
    return TimeIndex(horizon_steps)


def build_problem(cfg: Dict[str, Any]) -> Problem:
    """
    Build a Problem for the warehouse_fleet domain.

    `cfg` should already be normalized such that:
      cfg["domain"] == "warehouse_fleet"
      cfg["warehouse_fleet"] exists
    """
    if "warehouse_fleet" not in cfg:
        raise KeyError("Config missing required key: 'warehouse_fleet'")

    wf_cfg = cfg["warehouse_fleet"]
    time = _make_time_index(wf_cfg)

    # ---- Hook closures ----
    def simulate_fn(decision: Dict[str, Any]) -> Dict[str, Any]:
        # mobility.simulate should be deterministic given cfg + decision (+ optional seed)
        return wf_mobility.simulate(wf_cfg, decision, time)

    def constraints_fn(traces: Dict[str, Any], decision: Dict[str, Any]):
        # constraints.evaluate returns Sequence[ConstraintResult]
        return wf_constraints.evaluate(wf_cfg, traces, decision)

    def objective_fn(traces: Dict[str, Any], cons, decision: Dict[str, Any]):
        # objective.evaluate returns ObjectiveResult
        return wf_objective.evaluate(wf_cfg, traces, cons, decision)

    # ---- Optional helpers (sampling + repair) ----
    def sample_decision_fn(seed: int) -> Dict[str, Any]:
        # tasks.sample_decision should return a valid Decision dict.
        return wf_tasks.sample_decision(wf_cfg, seed)

    def repair_decision_fn(decision: Dict[str, Any]) -> Dict[str, Any]:
        # Optional: if tasks.py exposes repair_decision, use it;
        # otherwise do a tiny safety check + pass-through.
        repair = getattr(wf_tasks, "repair_decision", None)
        if callable(repair):
            return repair(wf_cfg, decision)  # type: ignore[return-value]

        # Minimal sanity check (won't fully validate correctness).
        if not isinstance(decision, dict):
            raise TypeError("Decision must be a dict")
        if "assignments" not in decision:
            # Let planners that use other decision formats work, but warn via error
            # only if you want strictness. For now, accept and let downstream fail.
            return decision
        if not isinstance(decision["assignments"], list):
            raise TypeError("Decision['assignments'] must be a list[list[int]]")
        return decision

    # ---- Construct Problem ----
    return Problem(
        name="warehouse_fleet",
        time=time,
        simulate_fn=simulate_fn,
        constraints_fn=constraints_fn,
        objective_fn=objective_fn,
        sample_decision_fn=sample_decision_fn,
        repair_decision_fn=repair_decision_fn,
        metadata={
            "domain": "warehouse_fleet",
            "horizon_steps": int(wf_cfg.get("horizon_steps", 200)),
            "grid": wf_cfg.get("grid", {}),
            "robots": wf_cfg.get("robots", {}),
            "tasks": wf_cfg.get("tasks", {}),
        },
    )
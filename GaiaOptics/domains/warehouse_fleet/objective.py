# gaiaoptics/domains/warehouse_fleet/objective.py
"""
Objective function for the Warehouse Fleet domain.

Implements:
  objective(traces, constraints, decision) -> ObjectiveResult

Design goals (accelerator-friendly, interpretable):
  - Primary objective: minimize total fleet energy usage
  - Optional secondary terms:
      * makespan proxy (total distance)
      * soft constraint penalties
  - Do NOT handle hard feasibility here (core feasibility is handled elsewhere)

Score convention:
  Lower score is better.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Sequence

from gaiaoptics.core.types import ConstraintResult, ObjectiveResult, Severity


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def evaluate(
    cfg: Dict[str, Any],
    traces: Dict[str, Any],
    constraints: Sequence[ConstraintResult],
    decision: Dict[str, Any],
) -> ObjectiveResult:
    """
    Compute scalar score + metrics dict.

    Score = energy_used
            + alpha * distance_total
            + beta * soft_constraint_penalties
    """
    # Primary term: energy
    energy = _as_float(traces.get("energy_used", 0.0), 0.0)

    # Secondary: distance proxy
    distance = _as_float(traces.get("distance_total", 0.0), 0.0)

    # Config weights (domain-local; safe if absent)
    obj_cfg = cfg.get("objective", {})
    if not isinstance(obj_cfg, dict):
        obj_cfg = {}

    alpha = _as_float(obj_cfg.get("distance_weight", 0.0), 0.0)      # default ignore distance
    beta = _as_float(obj_cfg.get("soft_penalty_weight", 10.0), 10.0) # default penalize soft violations

    # Soft constraint penalties (sum of magnitudes of negative margins)
    soft_penalty = 0.0
    for c in constraints:
        if c.severity == Severity.SOFT:
            margin = _as_float(getattr(c, "margin", 0.0), 0.0)
            if margin < 0.0:
                soft_penalty += -margin

    score = energy + alpha * distance + beta * soft_penalty

    metrics = {
        "energy_used": float(energy),
        "distance_total": float(distance),
        "soft_penalty": float(soft_penalty),
        "tasks_completed": _as_int(traces.get("tasks_completed", 0), 0),
        "task_completion_rate": _as_float(traces.get("task_completion_rate", 0.0), 0.0),
        "robot_battery_min": _as_float(traces.get("robot_battery_min", 0.0), 0.0),
        "score": float(score),
    }

    # Support different ObjectiveResult schemas across versions
    result_kwargs: Dict[str, Any] = {"score": float(score)}
    ctor_params = inspect.signature(ObjectiveResult).parameters
    for field_name in ("metrics", "details", "meta", "components"):
        if field_name in ctor_params:
            result_kwargs[field_name] = metrics
            break

    return ObjectiveResult(**result_kwargs)
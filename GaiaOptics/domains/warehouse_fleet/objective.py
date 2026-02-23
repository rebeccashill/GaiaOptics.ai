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
  - Do NOT handle hard feasibility here (core FeasibilityReport already does that)

Score convention:
  Lower score is better.
"""

from __future__ import annotations

import inspect
import inspect
from typing import Any, Dict, Sequence

from gaiaoptics.core.types import ConstraintResult, ObjectiveResult, Severity


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

    # ----------------------------
    # Primary term: energy
    # ----------------------------
    energy = float(traces.get("energy_used", 0.0))

    # ----------------------------
    # Secondary: distance proxy (optional)
    # ----------------------------
    distance = float(traces.get("distance_total", 0.0))

    obj_cfg = cfg.get("objective", {})
    alpha = float(obj_cfg.get("distance_weight", 0.0))  # default: ignore distance
    beta = float(obj_cfg.get("soft_penalty_weight", 10.0))

    # ----------------------------
    # Soft constraint penalties
    # ----------------------------
    soft_penalty = 0.0
    for c in constraints:
        if c.severity == Severity.SOFT and c.margin < 0.0:
            # margin < 0 means violation
            soft_penalty += -float(c.margin)

    # ----------------------------
    # Final score
    # ----------------------------
    score = energy + alpha * distance + beta * soft_penalty

    metrics = {
        # Core metric (what you’ll highlight in accelerator narrative)
        "energy_used": energy,
        # Supporting metrics
        "distance_total": distance,
        "soft_penalty": soft_penalty,
        "score": float(score),
        "tasks_completed": int(traces.get("tasks_completed", 0)),
        "task_completion_rate": float(traces.get("task_completion_rate", 0.0)),
        "robot_battery_min": float(traces.get("robot_battery_min", 0.0)),
    }

    result_kwargs: Dict[str, Any] = {"score": float(score)}
    ctor_params = inspect.signature(ObjectiveResult).parameters

    # Support different ObjectiveResult schemas across versions
    for field_name in ("metrics", "details", "meta"):
        if field_name in ctor_params:
            result_kwargs[field_name] = metrics
            break

    return ObjectiveResult(**result_kwargs)
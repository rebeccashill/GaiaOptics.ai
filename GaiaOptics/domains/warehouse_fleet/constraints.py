# gaiaoptics/domains/warehouse_fleet/constraints.py
"""
Constraint evaluation for the Warehouse Fleet domain.

This module implements the `constraints_fn` hook surface:
  constraints(traces, decision) -> Sequence[ConstraintResult]

V1 constraints (accelerator-friendly, minimal, interpretable):
  HARD:
    - battery_nonnegative: min robot battery >= 0
    - all_tasks_completed: completed == total (or completed - total >= 0 margin)
    - assignments_shape: decision has correct dimensions (robots count)
  SOFT (optional, only if you want):
    - unique_task_assignment: each task assigned exactly once

Notes:
- Margins should be >= 0 when satisfied.
- Keep details small; reporting will surface names + margins + severity.
"""

from __future__ import annotations

from typing import Any, Dict, List

from gaiaoptics.core.types import ConstraintResult, Severity


def _count_assigned_tasks(assignments: Any) -> Dict[str, Any]:
    """
    Returns:
      {
        "task_counts": dict[int, int],
        "total_mentions": int,
        "duplicates": int,    # number of task ids with count > 1
        "missing": int,       # number of task ids with count == 0 (requires n_tasks)
      }
    """
    task_counts: Dict[int, int] = {}
    total_mentions = 0
    if isinstance(assignments, list):
        for seq in assignments:
            if not isinstance(seq, list):
                continue
            for t in seq:
                try:
                    tid = int(t)
                except Exception:  # noqa: BLE001
                    continue
                task_counts[tid] = task_counts.get(tid, 0) + 1
                total_mentions += 1
    duplicates = sum(1 for _, c in task_counts.items() if c > 1)
    return {"task_counts": task_counts, "total_mentions": total_mentions, "duplicates": duplicates}


def evaluate(cfg: Dict[str, Any], traces: Dict[str, Any], decision: Dict[str, Any]) -> List[ConstraintResult]:
    robots_cfg = cfg.get("robots", {})
    tasks_cfg = cfg.get("tasks", {})

    n_robots = int(robots_cfg.get("count", 0))
    n_tasks = int(tasks_cfg.get("count", 0))

    cons: List[ConstraintResult] = []

    # ----------------------------
    # HARD: battery must stay >= 0
    # ----------------------------
    min_batt = float(traces.get("robot_battery_min", 0.0))
    cons.append(
        ConstraintResult(
            name="battery_nonnegative",
            severity=Severity.HARD,
            margin=min_batt,          # satisfied if >= 0
            details={},
        )
    )

    # ----------------------------
    # HARD: all tasks completed
    # ----------------------------
    completed = int(traces.get("tasks_completed", 0))
    # Margin >= 0 means completed >= n_tasks
    cons.append(
        ConstraintResult(
            name="all_tasks_completed",
            severity=Severity.HARD,
            margin=float(completed - n_tasks),
            details={"total_tasks": int(n_tasks)},
        )
    )

    # ----------------------------
    # HARD: assignments have correct shape (robots count)
    # ----------------------------
    assignments = decision.get("assignments", None)

    # Margin scheme:
    # - if missing assignments, margin = -1
    # - else margin = 0 if correct length, negative absolute difference otherwise
    if assignments is None:
        shape_margin = -1.0
        shape_value = 0.0
    elif not isinstance(assignments, list):
        shape_margin = -1.0
        shape_value = 0.0
    else:
        shape_value = float(len(assignments))
        shape_margin = -float(abs(len(assignments) - n_robots))

    cons.append(
        ConstraintResult(
            name="assignments_shape",
            severity=Severity.HARD,
            margin=float(shape_margin),  # satisfied if == 0 (we treat 0 as ok)
            details={"expected_robots": int(n_robots)},
        )
    )

    # ----------------------------
    # SOFT: unique task assignment (optional, helpful for cleaner solutions)
    # ----------------------------
    # Many early sims allow duplicates and still "complete" tasks once; this soft constraint
    # nudges the planner to assign each task exactly once.
    if isinstance(assignments, list) and n_tasks > 0:
        stats = _count_assigned_tasks(assignments)
        task_counts: Dict[int, int] = stats["task_counts"]

        missing = sum(1 for tid in range(n_tasks) if task_counts.get(tid, 0) == 0)
        duplicates = int(stats["duplicates"])

        # Perfect uniqueness: missing==0 and duplicates==0
        # Margin >= 0 means no missing and no duplicates.
        # We'll encode violations as -(missing + duplicates).
        uniq_margin = -float(missing + duplicates)

        cons.append(
            ConstraintResult(
                name="unique_task_assignment",
                severity=Severity.SOFT,
                margin=float(uniq_margin),
                details={"missing": int(missing), "duplicates": int(duplicates)},
            )
        )

    return cons
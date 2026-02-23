# gaiaoptics/domains/warehouse_fleet/mobility.py
"""
Mobility + energy model for the Warehouse Fleet domain.

This module implements the deterministic "physics" used by `simulate_fn`.

V1 design goals:
- Deterministic, simple, easy to reason about
- Energy is proportional to Manhattan distance traveled on a grid
- No charging, no congestion, no obstacles (add later without changing core hooks)

Config expectations (warehouse_fleet section):
  grid:
    width: int
    height: int
  robots:
    count: int
    battery_capacity: float
    energy_per_step: float
    starts: optional list[[x,y]] OR "random"
  tasks:
    count: int
    locations: optional list[[x,y]] OR "random"
  seed: optional int
  horizon_steps: int  (used for TimeIndex length)

Decision expectations (v1):
  decision = {
    "assignments": list[list[int]]  # tasks per robot, in visit order
  }

Traces returned (minimum, stable):
  traces = {
    "energy_used": float,
    "distance_total": float,
    "tasks_completed": int,
    "task_completion_rate": float,
    "robot_battery_min": float,
    "robot_energy_used": list[float],
    "robot_distance": list[float],
    "robot_final_pos": list[[int,int]],
    "robot_final_battery": list[float],
  }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import random

from gaiaoptics.core.types import TimeIndex


Coord = Tuple[int, int]


@dataclass(frozen=True)
class Task:
    id: int
    loc: Coord


def _require_int(d: Dict[str, Any], key: str) -> int:
    v = d.get(key, None)
    if v is None:
        raise KeyError(f"Missing required key: {key}")
    try:
        return int(v)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"Expected int for '{key}', got {type(v).__name__}") from e


def _require_float(d: Dict[str, Any], key: str) -> float:
    v = d.get(key, None)
    if v is None:
        raise KeyError(f"Missing required key: {key}")
    try:
        return float(v)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"Expected float for '{key}', got {type(v).__name__}") from e


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _in_bounds(p: Coord, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _parse_or_generate_locations(
    *,
    kind: str,
    count: int,
    width: int,
    height: int,
    seed: int,
    spec: Any,
) -> List[Coord]:
    """
    spec can be:
      - list of [x,y] coords
      - "random"
      - None (defaults to random)
    """
    rng = random.Random(seed)
    if spec is None or spec == "random":
        locs: List[Coord] = []
        for _ in range(count):
            locs.append((rng.randrange(width), rng.randrange(height)))
        return locs

    if isinstance(spec, list):
        locs2: List[Coord] = []
        for i, item in enumerate(spec):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise TypeError(f"{kind}.locations[{i}] must be [x,y]")
            x, y = int(item[0]), int(item[1])
            if not _in_bounds((x, y), width, height):
                raise ValueError(f"{kind}.locations[{i}] out of bounds: {(x, y)}")
            locs2.append((x, y))
        if len(locs2) != count:
            raise ValueError(f"{kind}.count={count} but provided {len(locs2)} {kind}.locations")
        return locs2

    raise TypeError(f"{kind}.locations must be a list[[x,y]] or 'random' (got {type(spec).__name__})")


def _get_tasks(cfg: Dict[str, Any], width: int, height: int) -> List[Task]:
    tasks_cfg = cfg.get("tasks", {})
    n_tasks = _require_int(tasks_cfg, "count")
    base_seed = int(cfg.get("seed", 0))

    loc_spec = tasks_cfg.get("locations", None)
    locs = _parse_or_generate_locations(
        kind="tasks",
        count=n_tasks,
        width=width,
        height=height,
        seed=base_seed + 1000,  # deterministic separation from robots
        spec=loc_spec,
    )
    return [Task(id=i, loc=locs[i]) for i in range(n_tasks)]


def _get_robot_starts(cfg: Dict[str, Any], width: int, height: int) -> List[Coord]:
    robots_cfg = cfg.get("robots", {})
    n_robots = _require_int(robots_cfg, "count")
    base_seed = int(cfg.get("seed", 0))

    starts_spec = robots_cfg.get("starts", None)
    starts = _parse_or_generate_locations(
        kind="robots",
        count=n_robots,
        width=width,
        height=height,
        seed=base_seed + 2000,
        spec=starts_spec,
    )
    return starts


def _validate_assignments(assignments: Any, n_robots: int, n_tasks: int) -> List[List[int]]:
    if not isinstance(assignments, list):
        raise TypeError("Decision['assignments'] must be list[list[int]]")
    if len(assignments) != n_robots:
        raise ValueError(f"Decision['assignments'] must have length {n_robots} (got {len(assignments)})")

    out: List[List[int]] = []
    for r, seq in enumerate(assignments):
        if not isinstance(seq, list):
            raise TypeError(f"Decision['assignments'][{r}] must be a list of task ids")
        seq2: List[int] = []
        for t in seq:
            tid = int(t)
            if tid < 0 or tid >= n_tasks:
                raise ValueError(f"Task id out of range in assignments[{r}]: {tid} (n_tasks={n_tasks})")
            seq2.append(tid)
        out.append(seq2)
    return out


def simulate(cfg: Dict[str, Any], decision: Dict[str, Any], time: TimeIndex) -> Dict[str, Any]:
    """
    Deterministic fleet simulation.

    - Each robot starts at its start coord.
    - It visits its assigned tasks in the given order.
    - Travel distance is Manhattan distance.
    - Energy consumed = distance * energy_per_step.
    - Battery starts at battery_capacity and is reduced by energy consumed.
    - If a robot runs out of battery (battery < 0), we still compute the traces but track min battery.

    Note: horizon_steps is currently used only for TimeIndex length; we do not enforce a step-by-step
    time rollout in v1. That can be added later without changing the public traces keys.
    """
    grid_cfg = cfg.get("grid", {})
    width = _require_int(grid_cfg, "width")
    height = _require_int(grid_cfg, "height")

    robots_cfg = cfg.get("robots", {})
    n_robots = _require_int(robots_cfg, "count")
    battery_capacity = _require_float(robots_cfg, "battery_capacity")
    energy_per_step = _require_float(robots_cfg, "energy_per_step")

    if battery_capacity <= 0:
        raise ValueError("robots.battery_capacity must be > 0")
    if energy_per_step <= 0:
        raise ValueError("robots.energy_per_step must be > 0")

    tasks = _get_tasks(cfg, width, height)
    n_tasks = len(tasks)
    task_locs = [t.loc for t in tasks]

    starts = _get_robot_starts(cfg, width, height)
    assignments_raw = decision.get("assignments", None)
    if assignments_raw is None:
        raise KeyError("Decision missing required key: 'assignments'")
    assignments = _validate_assignments(assignments_raw, n_robots=n_robots, n_tasks=n_tasks)

    # Simulate each robot independently (v1).
    robot_energy: List[float] = [0.0 for _ in range(n_robots)]
    robot_dist: List[float] = [0.0 for _ in range(n_robots)]
    robot_batt: List[float] = [float(battery_capacity) for _ in range(n_robots)]
    robot_pos: List[Coord] = list(starts)

    visited: List[bool] = [False for _ in range(n_tasks)]
    # Note: If tasks appear multiple times across robots, we count completion once.
    # You can add a constraint later to enforce exactly-once assignment.

    min_battery = float(battery_capacity)
    total_energy = 0.0
    total_dist = 0.0

    for r in range(n_robots):
        pos = robot_pos[r]
        batt = robot_batt[r]

        for tid in assignments[r]:
            target = task_locs[tid]
            d = manhattan(pos, target)
            e = float(d) * float(energy_per_step)

            batt -= e
            robot_energy[r] += e
            robot_dist[r] += float(d)

            total_energy += e
            total_dist += float(d)

            pos = target
            visited[tid] = True

            if batt < min_battery:
                min_battery = batt

        robot_pos[r] = pos
        robot_batt[r] = batt

    tasks_completed = sum(1 for v in visited if v)
    completion_rate = float(tasks_completed) / float(n_tasks) if n_tasks > 0 else 1.0

    traces: Dict[str, Any] = {
        # Primary metrics
        "energy_used": float(total_energy),
        "distance_total": float(total_dist),
        "tasks_completed": int(tasks_completed),
        "task_completion_rate": float(completion_rate),
        "robot_battery_min": float(min_battery),
        # Per-robot outputs
        "robot_energy_used": [float(x) for x in robot_energy],
        "robot_distance": [float(x) for x in robot_dist],
        "robot_final_pos": [[int(p[0]), int(p[1])] for p in robot_pos],
        "robot_final_battery": [float(x) for x in robot_batt],
        # Helpful debug/context
        "n_robots": int(n_robots),
        "n_tasks": int(n_tasks),
        "grid": {"width": int(width), "height": int(height)},
        "starts": [[int(p[0]), int(p[1])] for p in starts],
        "tasks": [[int(loc[0]), int(loc[1])] for loc in task_locs],
    }
    return traces
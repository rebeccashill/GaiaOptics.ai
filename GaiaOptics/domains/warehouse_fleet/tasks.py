# gaiaoptics/domains/warehouse_fleet/tasks.py
"""
Task + robot initialization helpers for the Warehouse Fleet domain.

Primary responsibilities:
- Provide a `sample_decision(cfg, seed)` function for the Problem.sample_decision_fn
- (Optional) Provide `repair_decision(cfg, decision)` to sanitize planner outputs

Decision contract (v1):
  decision = {"assignments": list[list[int]]}
where assignments[r] is an ordered list of task IDs for robot r.

Config expectations (warehouse_fleet section):
  robots:
    count: int
  tasks:
    count: int
  sampling:
    strategy: "random" | "round_robin" | "greedy_nearest" (default: "random")
    # greedy_nearest uses generated locations; must be consistent with mobility.py seeds
  seed: optional int
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import random


Coord = Tuple[int, int]


# ----------------------------
# Internal helpers
# ----------------------------

def _n_robots(cfg: Dict[str, Any]) -> int:
    robots_cfg = cfg.get("robots", {})
    n = int(robots_cfg.get("count", 0))
    if n <= 0:
        raise ValueError("warehouse_fleet.robots.count must be > 0")
    return n


def _n_tasks(cfg: Dict[str, Any]) -> int:
    tasks_cfg = cfg.get("tasks", {})
    n = int(tasks_cfg.get("count", 0))
    if n < 0:
        raise ValueError("warehouse_fleet.tasks.count must be >= 0")
    return n


def _grid(cfg: Dict[str, Any]) -> Tuple[int, int]:
    grid_cfg = cfg.get("grid", {})
    w = int(grid_cfg.get("width", 0))
    h = int(grid_cfg.get("height", 0))
    if w <= 0 or h <= 0:
        raise ValueError("warehouse_fleet.grid.width and grid.height must be > 0")
    return w, h


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _generate_locations(kind: str, count: int, width: int, height: int, seed: int, spec: Any) -> List[Coord]:
    """
    Must mirror mobility.py generation so greedy sampling stays consistent.

    spec can be:
      - list of [x,y] coords
      - "random" or None
    """
    rng = random.Random(seed)

    if spec is None or spec == "random":
        return [(rng.randrange(width), rng.randrange(height)) for _ in range(count)]

    if isinstance(spec, list):
        locs: List[Coord] = []
        for i, item in enumerate(spec):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise TypeError(f"{kind}.locations[{i}] must be [x,y]")
            locs.append((int(item[0]), int(item[1])))
        if len(locs) != count:
            raise ValueError(f"{kind}.count={count} but provided {len(locs)} {kind}.locations")
        return locs

    raise TypeError(f"{kind}.locations must be list[[x,y]] or 'random' (got {type(spec).__name__})")


def _get_tasks_locations(cfg: Dict[str, Any]) -> List[Coord]:
    width, height = _grid(cfg)
    base_seed = int(cfg.get("seed", 0))
    n_tasks = _n_tasks(cfg)
    tasks_cfg = cfg.get("tasks", {})
    return _generate_locations("tasks", n_tasks, width, height, base_seed + 1000, tasks_cfg.get("locations", None))


def _get_robot_starts(cfg: Dict[str, Any]) -> List[Coord]:
    width, height = _grid(cfg)
    base_seed = int(cfg.get("seed", 0))
    n_robots = _n_robots(cfg)
    robots_cfg = cfg.get("robots", {})
    return _generate_locations("robots", n_robots, width, height, base_seed + 2000, robots_cfg.get("starts", None))


# ----------------------------
# Public API
# ----------------------------

def sample_decision(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    Produce a valid candidate decision.

    Strategies:
      - random: assign each task to a random robot; task order is insertion order
      - round_robin: assign tasks sequentially across robots
      - greedy_nearest: build per-robot routes by repeatedly taking nearest unassigned task
                        (simple heuristic baseline; still deterministic)
    """
    n_robots = _n_robots(cfg)
    n_tasks = _n_tasks(cfg)

    sampling_cfg = cfg.get("sampling", {})
    strategy = str(sampling_cfg.get("strategy", "random")).lower()

    # Deterministic RNG: combine domain seed + planner seed so the same scenario can
    # produce different candidate decisions while remaining reproducible.
    base_seed = int(cfg.get("seed", 0))
    rng = random.Random(base_seed * 1_000_003 + int(seed))

    assignments: List[List[int]] = [[] for _ in range(n_robots)]

    if n_tasks == 0:
        return {"assignments": assignments}

    if strategy == "random":
        for tid in range(n_tasks):
            r = rng.randrange(n_robots)
            assignments[r].append(tid)
        return {"assignments": assignments}

    if strategy == "round_robin":
        for tid in range(n_tasks):
            assignments[tid % n_robots].append(tid)
        return {"assignments": assignments}

    if strategy == "greedy_nearest":
        # Greedy baseline: each robot repeatedly grabs nearest unassigned task from its current position.
        task_locs = _get_tasks_locations(cfg)
        robot_pos = _get_robot_starts(cfg)

        unassigned = set(range(n_tasks))

        # Iterate robots in a loop, each taking one nearest task at a time
        # until all tasks are assigned.
        while unassigned:
            progressed = False
            for r in range(n_robots):
                if not unassigned:
                    break
                pos = robot_pos[r]

                # pick nearest remaining task (tie-break deterministically by task id)
                best_tid = min(unassigned, key=lambda tid: (_manhattan(pos, task_locs[tid]), tid))
                assignments[r].append(best_tid)
                unassigned.remove(best_tid)

                robot_pos[r] = task_locs[best_tid]
                progressed = True

            if not progressed:
                # should never happen, but keep safe
                break

        return {"assignments": assignments}

    raise ValueError(f"Unknown warehouse_fleet.sampling.strategy: {strategy}")


def repair_decision(cfg: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional repair hook:
    - Ensures assignments has correct outer length (robots count)
    - Drops out-of-range task IDs
    - (Optionally) can enforce each task appears at most once (soft by default in constraints.py)

    This is intentionally conservative: it avoids "inventing" missing tasks, because your
    planner might want freedom to handle partials. Feasibility constraints will capture it.
    """
    n_robots = _n_robots(cfg)
    n_tasks = _n_tasks(cfg)

    if not isinstance(decision, dict):
        raise TypeError("Decision must be a dict")

    assignments = decision.get("assignments", None)
    if assignments is None:
        # Initialize with empty lists so the decision is always simulate-able.
        out = dict(decision)
        out["assignments"] = [[] for _ in range(n_robots)]
        return out

    if not isinstance(assignments, list):
        raise TypeError("Decision['assignments'] must be list[list[int]]")

    # Fix outer length
    fixed: List[List[int]] = []
    for r in range(n_robots):
        if r < len(assignments) and isinstance(assignments[r], list):
            seq = []
            for t in assignments[r]:
                try:
                    tid = int(t)
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= tid < n_tasks:
                    seq.append(tid)
            fixed.append(seq)
        else:
            fixed.append([])

    # Keep any extra keys untouched
    out = dict(decision)
    out["assignments"] = fixed
    return out
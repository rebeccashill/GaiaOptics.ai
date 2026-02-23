# gaiaoptics/core/planner.py
"""
Domain-agnostic planner (search/heuristics + restarts + acceptance rules).

This is intentionally simple to start:
- Uses Problem.sample_decision(seed) unless a sampler is passed in.
- Evaluates candidates with Problem.evaluate()
- Tracks best feasible, and best overall (for debugging)
- Supports restarts and a soft "accept_worse_prob" rule to avoid getting stuck.

Score convention:
- Lower is better (minimization). If you want maximize, negate in the objective_fn.

Feasibility convention:
- A candidate is feasible iff all HARD constraints have margin >= 0.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from gaiaoptics.core.problem import EvalResult, Problem, safe_evaluate
from gaiaoptics.core.types import (
    FeasibilityReport,
    ObjectiveResult,
    SolveStatus,
    Solution,
    Severity,
    to_jsonable,
)

Decision = Dict[str, Any]
SamplerFn = Callable[[Problem, int], Decision]


@dataclass
class PlannerConfig:
    """
    Basic planner configuration.

    iterations:
      total number of candidate evaluations across all restarts.

    restarts:
      number of independent "runs" (with different seed offsets). iterations are
      divided approximately evenly across restarts.

    accept_worse_prob:
      if in (0,1], the planner will sometimes accept a worse candidate as the
      current state (not as best), enabling a random-walk style exploration.

    seed:
      random seed for reproducibility.
    """
    iterations: int = 500
    restarts: int = 1
    accept_worse_prob: float = 0.0
    seed: int = 0

    # Optional behavior
    stop_on_feasible: bool = False   # stop when first feasible is found
    stop_on_score: Optional[float] = None  # stop if best_feasible_score <= stop_on_score

    # Debug/reporting knobs
    keep_best_infeasible: bool = True


@dataclass
class PlannerStats:
    evaluations: int = 0
    feasible_found: int = 0
    errors: int = 0
    best_feasible_score: Optional[float] = None
    best_infeasible_score: Optional[float] = None
    worst_hard_margin_best_feasible: Optional[float] = None
    worst_hard_constraint_best_feasible: Optional[str] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


def default_sampler(problem: Problem, seed: int) -> Decision:
    """
    Default candidate generator: delegates to Problem.sample_decision(seed).
    """
    return problem.sample_decision(seed)


def _is_better(a: EvalResult, b: EvalResult) -> bool:
    """
    Compare two evaluated candidates by objective score (lower is better).
    """
    return a.score < b.score


def _accept_worse(rng: random.Random, prob: float) -> bool:
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    return rng.random() < prob


def plan(
    problem: Problem,
    cfg: Optional[PlannerConfig] = None,
    sampler: Optional[SamplerFn] = None,
    label: str = "planner",
) -> Tuple[Solution, PlannerStats]:
    """
    Run the planner and return (best_solution, stats).

    The returned Solution is always populated. If no feasible solution is found:
      - status = NO_FEASIBLE
      - feasibility.feasible = False
      - contents represent the best infeasible candidate (if keep_best_infeasible)
        or a minimal empty placeholder otherwise.
    """
    cfg = cfg or PlannerConfig()
    sampler = sampler or default_sampler

    if cfg.iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {cfg.iterations}")
    if cfg.restarts <= 0:
        raise ValueError(f"restarts must be > 0, got {cfg.restarts}")
    if not (0.0 <= cfg.accept_worse_prob <= 1.0):
        raise ValueError("accept_worse_prob must be within [0, 1]")

    rng = random.Random(cfg.seed)
    t0 = time.time()

    stats = PlannerStats()

    best_feasible: Optional[EvalResult] = None
    best_infeasible: Optional[EvalResult] = None

    # A "current" state for random-walk acceptance, per restart
    current: Optional[EvalResult] = None

    # Divide iterations across restarts (last restart gets the remainder)
    base = cfg.iterations // cfg.restarts
    rem = cfg.iterations % cfg.restarts
    per_restart = [base + (1 if i < rem else 0) for i in range(cfg.restarts)]

    eval_index = 0

    for r in range(cfg.restarts):
        current = None
        # Make restart-specific seed offset deterministic but separated
        restart_seed_offset = (r + 1) * 10_000

        for i in range(per_restart[r]):
            seed_i = cfg.seed + restart_seed_offset + i
            decision = sampler(problem, seed_i)

            er, err = safe_evaluate(problem, decision)
            stats.evaluations += 1
            eval_index += 1

            if er is None:
                stats.errors += 1
                stats.last_error = err
                continue

            # Update best infeasible (if enabled) before feasibility check
            if (not er.feasible) and cfg.keep_best_infeasible:
                if best_infeasible is None or _is_better(er, best_infeasible):
                    best_infeasible = er
                    stats.best_infeasible_score = er.score

            if er.feasible:
                stats.feasible_found += 1
                if best_feasible is None or _is_better(er, best_feasible):
                    best_feasible = er
                    stats.best_feasible_score = er.score
                    stats.worst_hard_margin_best_feasible = er.worst_hard_margin
                    stats.worst_hard_constraint_best_feasible = er.worst_hard_constraint

                    # Optional stopping conditions
                    if cfg.stop_on_score is not None and er.score <= cfg.stop_on_score:
                        break
                    if cfg.stop_on_feasible:
                        break

            # Random-walk acceptance rule:
            # maintain a "current" that can move even if not best, to explore.
            if current is None:
                current = er
            else:
                if _is_better(er, current):
                    current = er
                else:
                    # accept worse sometimes
                    if _accept_worse(rng, cfg.accept_worse_prob):
                        current = er

        # If we broke early due to stop condition, exit outer loop too
        if cfg.stop_on_feasible and best_feasible is not None:
            break
        if cfg.stop_on_score is not None and best_feasible is not None and best_feasible.score <= cfg.stop_on_score:
            break

    runtime = time.time() - t0

    # Choose returned solution:
    chosen: Optional[EvalResult]
    status: SolveStatus

    if best_feasible is not None:
        chosen = best_feasible
        status = SolveStatus.OK
    else:
        chosen = best_infeasible if cfg.keep_best_infeasible else None
        status = SolveStatus.NO_FEASIBLE

    if chosen is None:
        # Minimal placeholder if nothing evaluated successfully
        feasibility = FeasibilityReport(
            feasible=False,
            worst_hard_margin=None,
            worst_hard_constraint=None,
            hard_violations=0,
            soft_violations=0,
            constraints=(),
        )
        objective = ObjectiveResult(score=float("inf"), components={"reason": float("inf")})
        sol = Solution(
            label=label,
            decision={},
            traces={},
            constraints=(),
            objective=objective,
            feasibility=feasibility,
            status=status,
            runtime_sec=runtime,
            iterations=stats.evaluations,
            restarts=cfg.restarts,
            seed=cfg.seed,
            notes={"planner_stats": stats.to_dict()},
        )
        return sol, stats

    sol = Solution(
        label=label,
        decision=chosen.decision,
        traces=chosen.traces,
        constraints=tuple(chosen.constraints),
        objective=chosen.objective,
        feasibility=chosen.feasibility,
        status=status,
        runtime_sec=runtime,
        iterations=stats.evaluations,
        restarts=cfg.restarts,
        seed=cfg.seed,
        notes={"planner_stats": stats.to_dict()},
    )
    return sol, stats
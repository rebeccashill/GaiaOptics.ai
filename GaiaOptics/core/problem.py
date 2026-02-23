# gaiaoptics/core/problem.py
"""
Domain hook container + evaluation pipeline.

A Problem is the domain-facing contract the core planner talks to.

Core idea:
  decision -> simulate -> constraints -> objective -> feasibility report (+ penalties)

Domains should provide small pure functions for:
  - sample/repair decisions (optional)
  - simulate(decision) -> traces
  - constraints(traces, decision) -> list[ConstraintResult]
  - objective(traces, constraints, decision) -> ObjectiveResult

This module stays domain-agnostic; it does not know what "battery" or "temperature" is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from gaiaoptics.core.types import (
    ConstraintResult,
    FeasibilityReport,
    ObjectiveResult,
    Severity,
    SolveStatus,
    TimeIndex,
)


# ----------------------------
# Hook type aliases
# ----------------------------

Decision = Dict[str, Any]
Traces = Dict[str, Any]

SimulateFn = Callable[[Decision], Traces]
ConstraintsFn = Callable[[Traces, Decision], Sequence[ConstraintResult]]
ObjectiveFn = Callable[[Traces, Sequence[ConstraintResult], Decision], ObjectiveResult]

# Optional helpers for planners/baselines
SampleDecisionFn = Callable[[int], Decision]  # input: seed -> decision
RepairDecisionFn = Callable[[Decision], Decision]


@dataclass(frozen=True)
class EvalResult:
    """
    Output of evaluating a single candidate decision.
    """
    decision: Decision
    traces: Traces
    constraints: Tuple[ConstraintResult, ...]
    objective: ObjectiveResult
    feasibility: FeasibilityReport

    @property
    def score(self) -> float:
        return float(self.objective.score)

    @property
    def feasible(self) -> bool:
        return bool(self.feasibility.feasible)

    @property
    def worst_hard_margin(self) -> Optional[float]:
        return self.feasibility.worst_hard_margin

    @property
    def worst_hard_constraint(self) -> Optional[str]:
        return self.feasibility.worst_hard_constraint


@dataclass
class Problem:
    """
    Core, domain-agnostic problem container.

    Required hooks:
      - simulate_fn
      - constraints_fn
      - objective_fn

    Optional hooks:
      - sample_decision_fn: for baselines / default planner candidate generation
      - repair_decision_fn: to fix up decisions before simulation (clamps, casts, etc.)
    """
    name: str
    time: TimeIndex

    simulate_fn: SimulateFn
    constraints_fn: ConstraintsFn
    objective_fn: ObjectiveFn

    sample_decision_fn: Optional[SampleDecisionFn] = None
    repair_decision_fn: Optional[RepairDecisionFn] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def repair(self, decision: Decision) -> Decision:
        if self.repair_decision_fn is None:
            return decision
        repaired = self.repair_decision_fn(decision)
        if not isinstance(repaired, dict):
            raise TypeError("repair_decision_fn must return a dict Decision")
        return repaired

        # --- Unified hook surface (required by tests) ---
    def simulate(self, decision):
        return self.simulate_fn(decision)

    def constraints(self, traces, decision):
        return self.constraints_fn(traces, decision)

    def objective(self, traces, cons, decision):
        return self.objective_fn(traces, cons, decision)

    def evaluate_constraints(self, traces: Traces, decision: Decision) -> Tuple[ConstraintResult, ...]:
        cons = list(self.constraints_fn(traces, decision))
        # Basic sanity checks so reporting doesn't break
        for c in cons:
            if not isinstance(c, ConstraintResult):
                raise TypeError("constraints_fn must return ConstraintResult objects")
        return tuple(cons)

    def evaluate_objective(
        self,
        traces: Traces,
        constraints: Sequence[ConstraintResult],
        decision: Decision,
    ) -> ObjectiveResult:
        obj = self.objective_fn(traces, constraints, decision)
        if not isinstance(obj, ObjectiveResult):
            raise TypeError("objective_fn must return an ObjectiveResult")
        return obj

    def evaluate(self, decision: Decision) -> EvalResult:
        """
        Fully evaluate a candidate decision.

        This is the core pipeline used by planners and baselines.
        """
        d = self.repair(decision)

        traces = self.simulate(d)
        constraints = self.evaluate_constraints(traces, d)
        feasibility = FeasibilityReport.from_constraints(constraints)
        objective = self.evaluate_objective(traces, constraints, d)

        return EvalResult(
            decision=d,
            traces=traces,
            constraints=constraints,
            objective=objective,
            feasibility=feasibility,
        )

    # ----------------------------
    # Convenience helpers for planners
    # ----------------------------

    def sample_decision(self, seed: int) -> Decision:
        """
        Optional: provide a domain default candidate generator.

        If you don't implement this, planners/baselines should provide their own.
        """
        if self.sample_decision_fn is None:
            raise RuntimeError(
                f"Problem '{self.name}' has no sample_decision_fn. "
                "Provide one in the domain mission builder or use a planner/baseline "
                "that doesn't require sampling from Problem."
            )
        d = self.sample_decision_fn(seed)
        if not isinstance(d, dict):
            raise TypeError("sample_decision_fn must return a dict Decision")
        return d

    def hard_constraint_summary(self, constraints: Sequence[ConstraintResult]) -> Dict[str, Any]:
        """
        Lightweight summary for reporting without importing reporting modules.
        """
        hard = [c for c in constraints if c.severity == Severity.HARD]
        worst = min(hard, key=lambda c: c.margin) if hard else None
        return {
            "hard_constraints": len(hard),
            "hard_violations": sum(1 for c in hard if c.margin < 0.0),
            "worst_hard_constraint": (worst.name if worst else None),
            "worst_hard_margin": (worst.margin if worst else None),
        }


# ----------------------------
# Safe evaluation wrapper (useful in planners)
# ----------------------------

def safe_evaluate(problem: Problem, decision: Decision) -> Tuple[Optional[EvalResult], Optional[str]]:
    """
    Evaluate a decision but return (None, error_message) rather than raising.
    Useful for long-running planners so one bad candidate doesn't kill the run.
    """
    try:
        return problem.evaluate(decision), None
    except Exception as e:  # noqa: BLE001 - intentionally broad for robustness
        return None, f"{type(e).__name__}: {e}"
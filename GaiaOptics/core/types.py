# gaiaoptics/core/types.py
"""
Shared, domain-agnostic types for GaiaOptics.

Keep this file small and stable: domains can evolve, but these types should remain
backwards-compatible as much as possible.

Design goals:
- Explicit HARD/SOFT constraint semantics
- Margin-first constraint reporting (positive = satisfied, negative = violated)
- Simple time index utility for horizon simulations
- JSON-friendly dataclasses
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# ----------------------------
# Enums
# ----------------------------

class Severity(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class SolveStatus(str, Enum):
    """High-level solve outcome."""
    OK = "OK"                      # Completed normally
    NO_FEASIBLE = "NO_FEASIBLE"    # Finished search but found no feasible solution
    ERROR = "ERROR"                # Exception / fatal error


# ----------------------------
# Time utilities
# ----------------------------

@dataclass(frozen=True)
class TimeIndex:
    """
    Discrete simulation timeline.

    - n_steps: number of discrete timesteps in the horizon
    - dt_hours: timestep size in hours (can be fractional, e.g., 0.25 for 15 min)
    """
    n_steps: int
    dt_hours: float = 1.0

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")
        if self.dt_hours <= 0:
            raise ValueError(f"dt_hours must be > 0, got {self.dt_hours}")

    def steps(self) -> range:
        return range(self.n_steps)

    def hours(self) -> List[float]:
        return [i * self.dt_hours for i in self.steps()]

    @property
    def horizon_hours(self) -> float:
        return self.n_steps * self.dt_hours


# ----------------------------
# Core results + reporting types
# ----------------------------

@dataclass(frozen=True)
class ConstraintResult:
    """
    Result of evaluating a single constraint.

    Margin convention:
      - margin >= 0  => satisfied
      - margin < 0   => violated

    For HARD constraints, any violation makes the solution infeasible.
    For SOFT constraints, violations contribute to penalty but can still be "feasible".
    """
    name: str
    severity: Severity
    margin: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def violated(self) -> bool:
        return self.margin < 0.0


@dataclass(frozen=True)
class FeasibilityReport:
    """
    Aggregate feasibility summary across constraints.
    """
    feasible: bool
    worst_hard_margin: Optional[float]
    worst_hard_constraint: Optional[str]
    hard_violations: int
    soft_violations: int
    # Useful for debugging/report generation
    constraints: Tuple[ConstraintResult, ...] = ()

    @staticmethod
    def from_constraints(constraints: Sequence[ConstraintResult]) -> "FeasibilityReport":
        hard = [c for c in constraints if c.severity == Severity.HARD]
        soft = [c for c in constraints if c.severity == Severity.SOFT]

        hard_violations = sum(1 for c in hard if c.violated)
        soft_violations = sum(1 for c in soft if c.violated)

        feasible = hard_violations == 0

        worst_hard: Optional[ConstraintResult] = None
        if hard:
            # "worst" is minimum margin (most negative if violated; smallest positive if all satisfied)
            worst_hard = min(hard, key=lambda c: c.margin)

        return FeasibilityReport(
            feasible=feasible,
            worst_hard_margin=(worst_hard.margin if worst_hard else None),
            worst_hard_constraint=(worst_hard.name if worst_hard else None),
            hard_violations=hard_violations,
            soft_violations=soft_violations,
            constraints=tuple(constraints),
        )


@dataclass(frozen=True)
class ObjectiveResult:
    """
    Objective evaluation output.

    score:
      - lower is better by default (recommended for minimization planners).
      - If you prefer higher-is-better, keep it consistent everywhere or negate.

    components:
      - breakdown for reporting (e.g., {"energy_cost": 123.4, "emissions": 50.1, "penalty": 1e6})
    """
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Solution:
    """
    Best solution found by the planner.

    decision:
      - domain-specific decision structure (often dict[str, Any] from decision_variables sample)
    traces:
      - domain-specific simulation traces (often dict[str, list[float]] or similar)
    constraints:
      - individual constraint results
    objective:
      - objective score + breakdown
    """
    label: str
    decision: Dict[str, Any]
    traces: Dict[str, Any]
    constraints: Tuple[ConstraintResult, ...]
    objective: ObjectiveResult
    feasibility: FeasibilityReport
    status: SolveStatus = SolveStatus.OK
    runtime_sec: Optional[float] = None
    iterations: Optional[int] = None
    restarts: Optional[int] = None
    seed: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(asdict(self))


# ----------------------------
# Trace helpers (optional but handy)
# ----------------------------

Number = Union[int, float]
Series = List[Number]
Traces = Dict[str, Union[Series, Number, str, Dict[str, Any]]]


# ----------------------------
# JSON helpers
# ----------------------------

def to_jsonable(obj: Any) -> Any:
    """
    Convert common Python objects (dataclasses, enums, tuples) into JSON-serializable
    forms. Safe to call on nested structures.
    """
    if obj is None:
        return None

    if isinstance(obj, Enum):
        return obj.value

    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # Basic scalar types pass through
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback: stringify unknown types (keeps exports robust)
    return str(obj)


# ----------------------------
# Units helpers (lightweight)
# ----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError(f"clamp bounds invalid: lo={lo} > hi={hi}")
    return lo if x < lo else hi if x > hi else x


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol
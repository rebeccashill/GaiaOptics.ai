# gaiaoptics/reporting/summary_metrics.py
"""
Minimal summary metric extraction for reports.

This module is domain-agnostic: it pulls values from Solution.objective.components
using common keys and falls back gracefully.

Recommended (microgrid) objective component keys:
- "energy_cost"
- "emissions_kg"
- "unserved_kwh"

You can standardize these across domains later (e.g., "energy_cost", "emissions_kg",
"unserved_kwh") to keep reports consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from gaiaoptics.core.types import Solution


@dataclass(frozen=True)
class SummaryMetrics:
    feasible: bool
    worst_hard_constraint: Optional[str]
    worst_hard_margin: Optional[float]

    energy_cost: Optional[float]
    emissions_kg: Optional[float]
    unserved_kwh: Optional[float]

    runtime_sec: Optional[float]
    iterations: Optional[int]


def _get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def extract_summary_metrics(sol: Solution) -> SummaryMetrics:
    comps = sol.objective.components or {}

    return SummaryMetrics(
        feasible=bool(sol.feasibility.feasible),
        worst_hard_constraint=sol.feasibility.worst_hard_constraint,
        worst_hard_margin=sol.feasibility.worst_hard_margin,
        energy_cost=_get_float(comps, "energy_cost"),
        emissions_kg=_get_float(comps, "emissions_kg"),
        unserved_kwh=_get_float(comps, "unserved_kwh"),
        runtime_sec=sol.runtime_sec,
        iterations=sol.iterations,
    )
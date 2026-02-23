# gaiaoptics/reporting/constraint_report.py
"""
Constraint report helpers.
"""

from __future__ import annotations

from typing import List, Sequence

from gaiaoptics.core.types import ConstraintResult


def worst_constraints(constraints: Sequence[ConstraintResult], k: int = 10) -> List[ConstraintResult]:
    """
    Return the k constraints with the worst (smallest) margins.

    Margin convention:
      - margin >= 0 => satisfied
      - margin < 0  => violated
    """
    kk = max(1, int(k))
    return sorted(list(constraints), key=lambda c: c.margin)[:kk]


def format_constraint_line(c: ConstraintResult) -> str:
    """
    Render a single constraint line for markdown output.
    """
    status = "VIOLATED" if c.violated else "ok"
    return f"- {c.severity.value:<4} | {status:<8} | margin={c.margin:.6g} | {c.name}"
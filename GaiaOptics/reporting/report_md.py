# gaiaoptics/reporting/report_md.py
"""
Write a usable report.md (no plots).

Required fields:
- feasible? (yes/no)
- worst hard constraint + margin
- total cost, emissions, unmet load
- runtime, iterations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from gaiaoptics.core.types import Solution, Severity
from gaiaoptics.reporting.constraint_report import worst_constraints
from gaiaoptics.reporting.summary_metrics import extract_summary_metrics


def _is_hard_constraint(c: Any) -> bool:
    """
    Prefer Severity-based classification (current contract).
    Fall back to legacy `hard: bool` if present.
    """
    sev = getattr(c, "severity", None)
    if sev is not None:
        return sev == Severity.HARD
    hard_flag = getattr(c, "hard", None)
    return bool(hard_flag) if hard_flag is not None else False


def _fmt_margin(margin: Any) -> str:
    if margin is None:
        return "n/a"
    try:
        x = float(margin)
    except Exception:
        return "n/a"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return f"{x:.6g}"


def _format_constraint_line(c: Any) -> str:
    name = str(getattr(c, "name", "constraint"))
    margin_txt = _fmt_margin(getattr(c, "margin", None))
    hard_txt = "hard" if _is_hard_constraint(c) else "soft"
    return f"- {name} ({hard_txt}): margin={margin_txt}"


def _fmt_opt(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return f"{x:.{digits}g}"


def write_report_md(
    out_path: Path,
    scenario: Dict[str, Any],
    solution: Solution,
    worst_k: int = 10,
) -> None:
    m = extract_summary_metrics(solution)

    scenario_name = str(scenario.get("name") or "scenario")
    domain = str(scenario.get("domain") or "unknown")

    lines: list[str] = []
    lines.append(f"# GaiaOptics Report — {scenario_name}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **domain:** {domain}")
    lines.append(f"- **status:** {solution.status.value}")
    lines.append(f"- **feasible:** {'yes' if m.feasible else 'no'}")
    lines.append(f"- **worst hard constraint:** {m.worst_hard_constraint or 'n/a'}")
    lines.append(f"- **worst hard margin:** {_fmt_opt(m.worst_hard_margin)}")
    lines.append("")

    lines.append("## Totals")
    lines.append(f"- **energy_cost:** {_fmt_opt(m.energy_cost)}")
    lines.append(f"- **emissions_kg:** {_fmt_opt(m.emissions_kg)}")
    lines.append(f"- **unserved_kwh:** {_fmt_opt(m.unserved_kwh)}")
    lines.append("")

    lines.append("## Run info")
    lines.append(f"- **runtime_sec:** {_fmt_opt(m.runtime_sec)}")
    lines.append(f"- **iterations:** {solution.iterations if solution.iterations is not None else 'n/a'}")
    lines.append(f"- **restarts:** {solution.restarts if solution.restarts is not None else 'n/a'}")
    lines.append("")

    lines.append(f"## Worst constraints (top {worst_k})")
    if solution.constraints:
        for c in worst_constraints(solution.constraints, k=worst_k):
            lines.append(_format_constraint_line(c))
    else:
        lines.append("- (no constraints reported)")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
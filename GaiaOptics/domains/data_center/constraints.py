# gaiaoptics/domains/data_center/constraints.py
"""
Data center constraints (Phase 2).

Keep it minimal and consistent with GaiaOptics core contracts:
- Return List[ConstraintResult]
- Use HARD constraints first
- Margins: positive/0 => satisfied; negative => violated

Starter constraints:
1) temp_max (HARD): room_temp_c[t] <= temp_max_c
2) cooling_power_cap (HARD): cooling_power_kw[t] <= p_max_kw
3) optional backlog_cap (HARD): backlog_kwh[t] <= backlog_cap_kwh (if modeled)

Notes:
- Humidity, power caps, SLA violation limits can be added later without changing interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from gaiaoptics.core.types import ConstraintResult, Severity


def _safe_series(traces: Dict[str, Any], key: str, n_steps: int) -> List[float]:
    v = traces.get(key)
    if not isinstance(v, list) or len(v) != n_steps or not all(isinstance(x, (int, float)) for x in v):
        raise ValueError(f"simulate() must return '{key}' as list[float] with length n_steps")
    return [float(x) for x in v]


@dataclass(frozen=True)
class DataCenterConstraintConfig:
    n_steps: int
    temp_max_c: float
    cooling_p_max_kw: float
    backlog_cap_kwh: Optional[float] = None


def evaluate_constraints_data_center(
    traces: Dict[str, Any],
    cfg: DataCenterConstraintConfig,
) -> List[ConstraintResult]:
    n = int(cfg.n_steps)

    room_temp_c = _safe_series(traces, "room_temp_c", n)
    cooling_power_kw = _safe_series(traces, "cooling_power_kw", n)

    out: List[ConstraintResult] = []

    # --- Temperature max ---
    # margin = min_t (temp_max - temp[t])
    temp_margin = min(float(cfg.temp_max_c) - float(t) for t in room_temp_c)
    out.append(
        ConstraintResult(
            name="temp_max",
            severity=Severity.HARD,
            margin=float(temp_margin),
            details={"temp_max_c": float(cfg.temp_max_c)},
        )
    )

    # --- Cooling power cap ---
    # margin = min_t (p_max - p[t])
    pmax = float(cfg.cooling_p_max_kw)
    p_margin = min(pmax - float(p) for p in cooling_power_kw)
    out.append(
        ConstraintResult(
            name="cooling_power_cap",
            severity=Severity.HARD,
            margin=float(p_margin),
            details={"p_max_kw": float(pmax)},
        )
    )

    # --- Optional backlog cap (if workload modeled) ---
    if cfg.backlog_cap_kwh is not None:
        backlog_kwh = _safe_series(traces, "backlog_kwh", n)
        cap = float(cfg.backlog_cap_kwh)
        backlog_margin = min(cap - float(b) for b in backlog_kwh)  # negative if backlog exceeds cap
        out.append(
            ConstraintResult(
                name="backlog_cap",
                severity=Severity.HARD,
                margin=float(backlog_margin),
                details={"backlog_cap_kwh": float(cap)},
            )
        )
    else:
        out.append(
            ConstraintResult(
                name="backlog_cap",
                severity=Severity.HARD,
                margin=float("inf"),
                details={"note": "no backlog cap configured"},
            )
        )

    return out
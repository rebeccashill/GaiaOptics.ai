# gaiaoptics/domains/microgrid/constraints.py
"""
Microgrid constraints.

HARD constraints (starter set):
- Battery SOC bounds: soc_min <= SOC(t) <= soc_max
- Load served: net demand must be met (no unserved load)

Assumptions in this starter model:
- There is always "grid import" available to serve remaining net load.
- Therefore, if grid is unconstrained, load served is always satisfied.
- To make "load served" meaningful as a HARD constraint, we optionally model
  a grid import limit (grid.p_max_import_kw). If not provided, we default to
  unconstrained grid and the constraint margin will be +inf.

If you prefer the "allow unmet load with heavy penalty" route, do it in the
objective instead and delete the HARD load_served constraint here.
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
class MicrogridConstraintConfig:
    """
    Optional constraint config.

    If p_max_import_kw is None: grid is considered unconstrained for load-served purposes.
    If provided: enforce grid_import_kw[t] <= p_max_import_kw as HARD.
    """
    soc_min_kwh: float
    soc_max_kwh: float
    p_max_import_kw: Optional[float] = None


def evaluate_constraints_microgrid(
    traces: Dict[str, Any],
    n_steps: int,
    cfg: MicrogridConstraintConfig,
) -> List[ConstraintResult]:
    soc = _safe_series(traces, "soc_kwh", n_steps)
    grid_import = _safe_series(traces, "grid_import_kw", n_steps)
    net_kw = _safe_series(traces, "net_kw", n_steps)

    # --- SOC bounds ---
    # margin for min: min_t (SOC - soc_min)
    min_soc_margin = min(s - cfg.soc_min_kwh for s in soc)
    # margin for max: min_t (soc_max - SOC)
    max_soc_margin = min(cfg.soc_max_kwh - s for s in soc)

    out: List[ConstraintResult] = [
        ConstraintResult(
            name="battery_soc_min",
            severity=Severity.HARD,
            margin=float(min_soc_margin),
            details={"soc_min_kwh": cfg.soc_min_kwh},
        ),
        ConstraintResult(
            name="battery_soc_max",
            severity=Severity.HARD,
            margin=float(max_soc_margin),
            details={"soc_max_kwh": cfg.soc_max_kwh},
        ),
    ]

    # --- Load served (implemented as "no unserved power") ---
    #
    # In this model, unserved load only happens if:
    #   net_kw[t] > 0 AND grid import is capped below net_kw[t].
    #
    # If no cap, load_served_margin is +inf (always satisfied).
    if cfg.p_max_import_kw is None:
        out.append(
            ConstraintResult(
                name="load_served",
                severity=Severity.HARD,
                margin=float("inf"),
                details={"note": "unconstrained_grid_import => always served"},
            )
        )
        return out

    cap = float(cfg.p_max_import_kw)

    # Unserved at t: max(0, net - cap)
    # Constraint wants unserved == 0 => margin = -max_unserved_kw
    max_unserved_kw = 0.0
    for t in range(n_steps):
        unserved = max(0.0, float(net_kw[t]) - cap)
        if unserved > max_unserved_kw:
            max_unserved_kw = unserved

    load_served_margin = -max_unserved_kw  # 0 if fully served, negative otherwise

    out.append(
        ConstraintResult(
            name="load_served",
            severity=Severity.HARD,
            margin=float(load_served_margin),
            details={"p_max_import_kw": cap, "max_unserved_kw": float(max_unserved_kw)},
        )
    )

    # Also enforce the import cap itself as a HARD constraint (useful for debugging)
    cap_margin = min(cap - float(p) for p in grid_import)  # negative if any import exceeds cap
    out.append(
        ConstraintResult(
            name="grid_import_cap",
            severity=Severity.HARD,
            margin=float(cap_margin),
            details={"p_max_import_kw": cap},
        )
    )

    return out
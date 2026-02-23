# gaiaoptics/domains/water_network/constraints.py
"""
Water network constraints (Phase 2).

Designed to pair with hydraulics.py + mission.py toy tank model.

Constraints (HARD):
- tank_level_min: tank_level_m3[t] >= tank_min_m3
- tank_level_max: tank_level_m3[t] <= tank_max_m3
- pump_power_cap: pump_power_kw[t] <= p_max_kw

Conventions:
- margin >= 0 means satisfied
- margin < 0 means violated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gaiaoptics.core.types import ConstraintResult, Severity


def _series(traces: Dict[str, Any], key: str, n_steps: int) -> List[float]:
    v = traces.get(key)
    if not isinstance(v, list) or len(v) != n_steps or not all(isinstance(x, (int, float)) for x in v):
        raise ValueError(f"simulate() must return '{key}' as list[float] with length n_steps")
    return [float(x) for x in v]


@dataclass(frozen=True)
class WaterNetworkConstraintConfig:
    n_steps: int
    tank_min_m3: float
    tank_max_m3: float
    pump_p_max_kw: float


def evaluate_constraints_water_network(
    traces: Dict[str, Any],
    cfg: WaterNetworkConstraintConfig,
) -> List[ConstraintResult]:
    n = int(cfg.n_steps)

    tank_level_m3 = _series(traces, "tank_level_m3", n)
    pump_power_kw = _series(traces, "pump_power_kw", n)

    out: List[ConstraintResult] = []

    # tank_level_min: margin = min_t(level - min)
    min_margin = min(float(x) - float(cfg.tank_min_m3) for x in tank_level_m3)
    out.append(
        ConstraintResult(
            name="tank_level_min",
            severity=Severity.HARD,
            margin=float(min_margin),
            details={"tank_min_m3": float(cfg.tank_min_m3)},
        )
    )

    # tank_level_max: margin = min_t(max - level)
    max_margin = min(float(cfg.tank_max_m3) - float(x) for x in tank_level_m3)
    out.append(
        ConstraintResult(
            name="tank_level_max",
            severity=Severity.HARD,
            margin=float(max_margin),
            details={"tank_max_m3": float(cfg.tank_max_m3)},
        )
    )

    # pump_power_cap: margin = min_t(p_max - p)
    pmax = float(cfg.pump_p_max_kw)
    p_margin = min(pmax - float(x) for x in pump_power_kw)
    out.append(
        ConstraintResult(
            name="pump_power_cap",
            severity=Severity.HARD,
            margin=float(p_margin),
            details={"p_max_kw": float(pmax)},
        )
    )

    return out
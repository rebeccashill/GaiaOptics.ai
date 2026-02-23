# gaiaoptics/domains/water_network/hydraulics.py
"""
Water network hydraulics (Phase 2 toy model).

This module provides a minimal, deterministic "hydraulics" layer for the tank + pump
toy domain. It is intentionally simple but structured so you can extend later to:
- multiple tanks / junctions / pipes
- head-loss curves (Hazen-Williams / Darcy-Weisbach)
- pump curves (flow vs head vs efficiency)
- pressure constraints at nodes
- energy intensity of pumping as a function of flow/head

For Phase 2, we treat the pump as:
  flow_in_m3ph = pump_power_kw * pump_flow_m3ph_per_kw

And the tank mass-balance:
  tank_next = tank + (flow_in - demand) * dt_hours

All functions are pure and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class PumpParams:
    """
    p_max_kw:
      Maximum pump electrical power.

    pump_flow_m3ph_per_kw:
      Linear mapping from electrical power to volumetric flow (m^3/hour per kW).
      This is a toy stand-in for pump curve + efficiency.
    """
    p_max_kw: float = 20.0
    pump_flow_m3ph_per_kw: float = 1.0


@dataclass(frozen=True)
class TankParams:
    """
    tank0_m3:
      Initial tank volume.

    tank_min_m3 / tank_max_m3:
      Bounds for feasibility.
    """
    tank0_m3: float = 50.0
    tank_min_m3: float = 0.0
    tank_max_m3: float = 100.0


def pump_flow_m3ph(pump_power_kw: float, params: PumpParams) -> float:
    """
    Convert pump electrical power to inflow rate (m^3/hour).
    """
    return float(pump_power_kw) * float(params.pump_flow_m3ph_per_kw)


def tank_step_m3(
    tank_level_m3: float,
    flow_in_m3ph: float,
    demand_m3ph: float,
    dt_hours: float,
) -> float:
    """
    Single-step tank mass balance.
    """
    return float(tank_level_m3) + (float(flow_in_m3ph) - float(demand_m3ph)) * float(dt_hours)


def simulate_tank_network(
    *,
    n_steps: int,
    dt_hours: float,
    tank0_m3: float,
    demand_m3ph: Sequence[float],
    pump_power_kw: Sequence[float],
    pump_params: PumpParams,
) -> dict:
    """
    Simulate tank + pump over horizon.

    Returns traces:
      tank_level_m3
      flow_in_m3ph
      flow_out_m3ph
      pump_power_kw
      dt_hours
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")
    if len(demand_m3ph) != n_steps:
        raise ValueError("demand_m3ph must have length n_steps")
    if len(pump_power_kw) != n_steps:
        raise ValueError("pump_power_kw must have length n_steps")

    tank = float(tank0_m3)

    tank_level: List[float] = []
    inflow: List[float] = []
    outflow: List[float] = []
    p_series: List[float] = []

    for t in range(n_steps):
        p_kw = float(pump_power_kw[t])
        d = float(demand_m3ph[t])

        q_in = pump_flow_m3ph(p_kw, pump_params)
        tank = tank_step_m3(tank, q_in, d, dt_hours)

        tank_level.append(float(tank))
        inflow.append(float(q_in))
        outflow.append(float(d))
        p_series.append(float(p_kw))

    return {
        "tank_level_m3": tank_level,
        "flow_in_m3ph": inflow,
        "flow_out_m3ph": outflow,
        "pump_power_kw": p_series,
        "demand_m3ph": [float(x) for x in demand_m3ph],
        "dt_hours": float(dt_hours),
    }
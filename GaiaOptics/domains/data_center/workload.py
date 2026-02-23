# gaiaoptics/domains/data_center/thermal.py
"""
Data center thermal model (toy but coherent).

This module intentionally stays minimal for Phase 2:
- Single-zone (room air) temperature state
- Heat inputs from IT load + (optional) ambient envelope gains
- Cooling removal proportional to electrical cooling power via COP

Why a separate module?
- Keeps mission.py readable and consistent with the "platform" story
- Makes it easy to later extend to multi-zone / humidity / economizer / CRAC dynamics

All functions are pure + deterministic (no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ThermalParams:
    """
    thermal_mass_kwh_per_c:
      Effective thermal capacity (kWh per °C). Larger => slower temperature change.

    ua_kw_per_c:
      Envelope heat transfer coefficient (kW per °C), applied when ambient > room.

    cop:
      Coefficient of performance for cooling (kW_removed per kW_electric).
    """

    thermal_mass_kwh_per_c: float = 50.0
    ua_kw_per_c: float = 2.0
    cop: float = 3.0


def envelope_heat_gain_kw(ua_kw_per_c: float, ambient_temp_c: float, room_temp_c: float) -> float:
    """
    Simple envelope heat gain model:
      gain = UA * max(0, ambient - room)

    This avoids negative "free cooling" effects in Phase 2; you can add economizer later.
    """
    return float(ua_kw_per_c) * max(0.0, float(ambient_temp_c) - float(room_temp_c))


def cooling_removed_kw(cooling_power_kw: float, cop: float) -> float:
    """
    Cooling removed (kW thermal) from electrical cooling power (kW) via COP.
    """
    return float(cooling_power_kw) * float(cop)


def step_room_temp_c(
    room_temp_c: float,
    ambient_temp_c: float,
    it_load_kw: float,
    cooling_power_kw: float,
    dt_hours: float,
    params: ThermalParams,
) -> Tuple[float, float, float]:
    """
    Advance the room temperature by one timestep.

    Returns:
      (next_room_temp_c, heat_in_kw, cooling_removed_kw)

    Dynamics:
      heat_in_kw = it_load_kw + UA * max(0, ambient - room)
      cooling_kw = cooling_power_kw * COP
      dT = (heat_in_kw - cooling_kw) * dt_hours / thermal_mass_kwh_per_c
    """
    env_kw = envelope_heat_gain_kw(params.ua_kw_per_c, ambient_temp_c, room_temp_c)
    heat_in_kw = float(it_load_kw) + float(env_kw)
    cool_kw = cooling_removed_kw(cooling_power_kw, params.cop)

    dtemp = (heat_in_kw - cool_kw) * float(dt_hours) / float(params.thermal_mass_kwh_per_c)
    next_temp = float(room_temp_c) + float(dtemp)
    return next_temp, float(heat_in_kw), float(cool_kw)


def simulate_single_zone(
    *,
    n_steps: int,
    dt_hours: float,
    temp0_c: float,
    ambient_temp_c: Sequence[float],
    it_load_kw: Sequence[float],
    cooling_power_kw: Sequence[float],
    params: ThermalParams,
) -> dict:
    """
    Simulate a single-zone data center over a horizon.

    Returns a dict of traces (all list[float] of length n_steps):
      room_temp_c
      heat_in_kw
      cooling_removed_kw
      cooling_power_kw
      ambient_temp_c
      it_load_kw
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")
    if len(ambient_temp_c) != n_steps:
        raise ValueError("ambient_temp_c must have length n_steps")
    if len(it_load_kw) != n_steps:
        raise ValueError("it_load_kw must have length n_steps")
    if len(cooling_power_kw) != n_steps:
        raise ValueError("cooling_power_kw must have length n_steps")

    temp = float(temp0_c)

    temps: List[float] = []
    heat_in: List[float] = []
    cool_out: List[float] = []

    for t in range(n_steps):
        temp, h_kw, c_kw = step_room_temp_c(
            room_temp_c=temp,
            ambient_temp_c=float(ambient_temp_c[t]),
            it_load_kw=float(it_load_kw[t]),
            cooling_power_kw=float(cooling_power_kw[t]),
            dt_hours=float(dt_hours),
            params=params,
        )
        temps.append(float(temp))
        heat_in.append(float(h_kw))
        cool_out.append(float(c_kw))

    return {
        "room_temp_c": temps,
        "heat_in_kw": heat_in,
        "cooling_removed_kw": cool_out,
        "cooling_power_kw": [float(x) for x in cooling_power_kw],
        "ambient_temp_c": [float(x) for x in ambient_temp_c],
        "it_load_kw": [float(x) for x in it_load_kw],
        "dt_hours": float(dt_hours),
    }
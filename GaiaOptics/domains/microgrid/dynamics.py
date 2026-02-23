# gaiaoptics/domains/microgrid/dynamics.py
"""
Microgrid dynamics (toy but coherent).

State variables:
- battery SOC (kWh)
- generator output (kW) [optional but supported]

Exogenous series (from YAML, provided to simulate()):
- PV (kW)
- load (kW)

Decision variables (per timestep):
- battery_power_kw[t]: (+) discharge to serve load, (-) charge from surplus/grid
- gen_power_kw[t]: generator output (kW)

Power balance (no export by default in this module):
  net_kw = load_kw - pv_kw - battery_power_kw - gen_power_kw
  grid_import_kw = max(0, net_kw)
If you want export later, carry net_kw through and allow negative grid.

SOC update (dt in hours):
  if batt >= 0 (discharge):  soc -= (batt/eta_discharge)*dt
  if batt < 0 (charge):      soc += (-batt*eta_charge)*dt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gaiaoptics.core.types import TimeIndex, clamp


@dataclass(frozen=True)
class BatteryParams:
    capacity_kwh: float
    soc0_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    p_max_kw: float
    eta_charge: float = 0.95
    eta_discharge: float = 0.95


@dataclass(frozen=True)
class GeneratorParams:
    """
    Very simple generator model.

    - p_min/p_max: output bounds
    - ramp_kw_per_step: max |p[t]-p[t-1]| per timestep (optional)
    - fuel_cost_per_kwh: used in objective (not here) but stored for convenience
    """
    enabled: bool = True
    p_min_kw: float = 0.0
    p_max_kw: float = 50.0
    ramp_kw_per_step: Optional[float] = None


@dataclass(frozen=True)
class MicrogridParams:
    time: TimeIndex
    battery: BatteryParams
    generator: GeneratorParams
    allow_export: bool = False  # if True, grid_import_kw can be negative (export)


def simulate_microgrid(
    params: MicrogridParams,
    load_kw: Sequence[float],
    pv_kw: Sequence[float],
    battery_power_kw: Sequence[float],
    gen_power_kw: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Simulate microgrid over the horizon.

    Returns traces:
      t
      load_kw
      pv_kw
      battery_power_kw (after clamp)
      gen_power_kw (after clamp; zeros if disabled)
      soc_kwh
      net_kw
      grid_import_kw
    """
    n = params.time.n_steps
    dt = params.time.dt_hours

    if len(load_kw) != n or len(pv_kw) != n:
        raise ValueError("load_kw and pv_kw must match horizon length")
    if len(battery_power_kw) != n:
        raise ValueError("battery_power_kw must match horizon length")
    if gen_power_kw is not None and len(gen_power_kw) != n:
        raise ValueError("gen_power_kw must match horizon length")

    batt = params.battery
    gen = params.generator

    soc = float(batt.soc0_kwh)

    soc_series: List[float] = []
    batt_series: List[float] = []
    gen_series: List[float] = []
    net_series: List[float] = []
    grid_series: List[float] = []

    prev_gen = 0.0

    for t in range(n):
        # Clamp battery power
        p_batt = clamp(float(battery_power_kw[t]), -batt.p_max_kw, batt.p_max_kw)

        # Generator (optional)
        if gen.enabled:
            raw = float(gen_power_kw[t]) if gen_power_kw is not None else 0.0
            p_gen = clamp(raw, gen.p_min_kw, gen.p_max_kw)

            # Ramp constraint (if specified) is not enforced here as HARD clamp,
            # but we can "physics-clamp" to keep sim stable.
            if gen.ramp_kw_per_step is not None:
                max_delta = float(gen.ramp_kw_per_step)
                p_gen = clamp(p_gen, prev_gen - max_delta, prev_gen + max_delta)

            prev_gen = p_gen
        else:
            p_gen = 0.0

        # Power balance
        net_kw = float(load_kw[t]) - float(pv_kw[t]) - p_batt - p_gen

        if params.allow_export:
            grid_import_kw = net_kw  # may be negative (export)
        else:
            grid_import_kw = max(0.0, net_kw)

        # SOC update
        if p_batt >= 0.0:
            soc -= (p_batt / max(batt.eta_discharge, 1e-9)) * dt
        else:
            soc += (-p_batt * batt.eta_charge) * dt

        # Record traces (SOC after update)
        batt_series.append(p_batt)
        gen_series.append(p_gen)
        net_series.append(net_kw)
        grid_series.append(grid_import_kw)
        soc_series.append(soc)

    return {
        "t": list(range(n)),
        "load_kw": [float(x) for x in load_kw],
        "pv_kw": [float(x) for x in pv_kw],
        "battery_power_kw": batt_series,
        "gen_power_kw": gen_series,
        "soc_kwh": soc_series,
        "net_kw": net_series,
        "grid_import_kw": grid_series,
        "dt_hours": float(dt),
    }


def build_params_from_config(cfg: Dict[str, Any]) -> Tuple[MicrogridParams, List[float], List[float]]:
    """
    Convenience helper to parse minimal config pieces required for dynamics.

    Returns:
      (params, load_kw, pv_kw)
    """
    horizon = cfg.get("horizon", {}) or {}
    n_steps = int(horizon.get("n_steps", 24))
    dt_hours = float(horizon.get("dt_hours", 1.0))
    time = TimeIndex(n_steps=n_steps, dt_hours=dt_hours)

    series = cfg.get("series", {}) or {}
    load_kw = series.get("load_kw")
    pv_kw = series.get("pv_kw")
    if not isinstance(load_kw, list) or not all(isinstance(x, (int, float)) for x in load_kw):
        raise ValueError("series.load_kw must be a list of numbers")
    if not isinstance(pv_kw, list) or not all(isinstance(x, (int, float)) for x in pv_kw):
        raise ValueError("series.pv_kw must be a list of numbers")
    if len(load_kw) != n_steps or len(pv_kw) != n_steps:
        raise ValueError("series.load_kw and series.pv_kw must match horizon.n_steps")

    bcfg = cfg.get("battery", {}) or {}
    cap = float(bcfg.get("capacity_kwh", 50.0))
    soc0 = float(bcfg.get("soc0_kwh", cap / 2.0))
    soc_min = float(bcfg.get("soc_min_kwh", 0.0))
    soc_max = float(bcfg.get("soc_max_kwh", cap))
    pmax = float(bcfg.get("p_max_kw", 25.0))
    eta_c = float(bcfg.get("eta_charge", 0.95))
    eta_d = float(bcfg.get("eta_discharge", 0.95))

    battery = BatteryParams(
        capacity_kwh=cap,
        soc0_kwh=soc0,
        soc_min_kwh=soc_min,
        soc_max_kwh=soc_max,
        p_max_kw=pmax,
        eta_charge=eta_c,
        eta_discharge=eta_d,
    )

    gcfg = cfg.get("generator", {}) or {}
    gen_enabled = bool(gcfg.get("enabled", True))
    generator = GeneratorParams(
        enabled=gen_enabled,
        p_min_kw=float(gcfg.get("p_min_kw", 0.0)),
        p_max_kw=float(gcfg.get("p_max_kw", 0.0 if not gen_enabled else 50.0)),
        ramp_kw_per_step=(float(gcfg["ramp_kw_per_step"]) if "ramp_kw_per_step" in gcfg else None),
    )

    grid = cfg.get("grid", {}) or {}
    allow_export = bool(grid.get("allow_export", False))

    params = MicrogridParams(time=time, battery=battery, generator=generator, allow_export=allow_export)
    return params, [float(x) for x in load_kw], [float(x) for x in pv_kw]
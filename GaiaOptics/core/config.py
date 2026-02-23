# gaiaoptics/core/config.py
from __future__ import annotations

from dataclasses import dataclass
from os import error
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import difflib

import yaml

from gaiaoptics.core.errors import ConfigError


CANONICAL_TOP_LEVEL_KEYS = ("scenario", "run", "planner", "objectives", "microgrid", "data_center", "water_network")

def _domain_from_cfg(cfg: Mapping[str, Any]) -> str | None:
    # Preferred schema
    scenario = cfg.get("scenario")
    if isinstance(scenario, Mapping):
        d = scenario.get("domain")
        if isinstance(d, Mapping):
            name = d.get("name")
            return name if isinstance(name, str) else None
        return d if isinstance(d, str) else None

    # Legacy schema: domain at top-level
    d = cfg.get("domain")
    if isinstance(d, Mapping):
        name = d.get("name")
        return name if isinstance(name, str) else None
    return d if isinstance(d, str) else None

def _as_dict(obj: Any, path: str) -> Dict[str, Any]:
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ConfigError(path, f"expected a mapping/object, got {type(obj).__name__}")
    return obj


def _require_str(d: Mapping[str, Any], key: str, path: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"{path}.{key}", "expected a non-empty string")
    return v.strip()


def _require_num(d: Mapping[str, Any], key: str, path: str) -> float:
    v = d.get(key)
    if not isinstance(v, (int, float)):
        raise ConfigError(f"{path}.{key}", f"expected a number, got {type(v).__name__}")
    return float(v)


def _suggest_key(bad_key: str, allowed: Tuple[str, ...]) -> Optional[str]:
    close = difflib.get_close_matches(bad_key, allowed, n=1, cutoff=0.7)
    return close[0] if close else None


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ConfigError("yaml", f"failed to parse YAML: {e}") from e
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("yaml", f"top-level YAML must be a mapping/object, got {type(raw).__name__}")
    return raw


def canonical_yaml_dump(data: Mapping[str, Any]) -> str:
    """
    Stable dump: sorted keys + deterministic formatting.
    """
    return yaml.safe_dump(
        data,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )

from typing import Any, Dict


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = parent.get(key)
    if v is None:
        parent[key] = {}
        return parent[key]
    if not isinstance(v, dict):
        raise ConfigError(key, f"expected '{key}' to be a mapping")
    return v


def _normalize_microgrid(mg: Dict[str, Any], run: Dict[str, Any]) -> None:
    # run defaults
    run.setdefault("horizon_hours", 24)
    run.setdefault("timestep_minutes", 60)

    mg.setdefault(
        "series",
        {"price_per_kwh": 0.20, "carbon_kg_per_kwh": 0.40},
    )
    mg.setdefault(
        "battery",
        {
            "capacity_kwh": 20.0,
            "soc0_kwh": 10.0,
            "p_charge_max_kw": 10.0,
            "p_discharge_max_kw": 10.0,
            "eta_charge": 0.95,
            "eta_discharge": 0.95,
        },
    )

def normalize_config(raw: Mapping[str, Any], *, source_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Accepts "messy but acceptable" YAML and returns a canonical dict:
      - scenario/run/planner/objectives/<domain>
      - defaults filled
      - derived fields computed (scenario.name, scenario.output_dir)
    """
    raw = _as_dict(raw, "yaml")

    # --- Gather legacy shortcuts ---
    # --- Gather legacy shortcuts ---
    scenario_name = raw.get("scenario_name") or raw.get("name")
    domain = _domain_from_cfg(raw)  # <-- ALWAYS returns str|None

    scenario = _as_dict(raw.get("scenario"), "scenario")
    run = _as_dict(raw.get("run"), "run")
    planner = _as_dict(raw.get("planner"), "planner")
    objectives = _as_dict(raw.get("objectives"), "objectives")

    # domain payloads (namespaced)
    microgrid = _as_dict(raw.get("microgrid"), "microgrid")
    data_center = _as_dict(raw.get("data_center"), "data_center")
    water_network = _as_dict(raw.get("water_network"), "water_network")

    # allow some older shapes like top-level "seed", "horizon_hours", etc.
    if "seed" in raw and "seed" not in scenario:
        scenario["seed"] = raw["seed"]
    if "horizon_hours" in raw and "horizon_hours" not in run:
        run["horizon_hours"] = raw["horizon_hours"]
    if "timestep_minutes" in raw and "timestep_minutes" not in run:
        run["timestep_minutes"] = raw["timestep_minutes"]

    # scenario defaults
    if "domain" not in scenario and domain:
        scenario["domain"] = domain
    if "name" not in scenario and scenario_name:
        scenario["name"] = scenario_name

    # derive scenario.name from filename if missing
    if "name" not in scenario:
        scenario["name"] = source_path.stem if source_path is not None else "scenario"

    # defaults
    scenario.setdefault("seed", 0)
    run.setdefault("horizon_hours", 24)
    run.setdefault("timestep_minutes", 60)

    planner.setdefault("name", "baseline_random")
    planner.setdefault("iterations", 200)
    planner.setdefault("restarts", 1)

    objectives.setdefault("cost_weight", 1.0)
    objectives.setdefault("emissions_weight", 1.0)

    # derived output dir (relative, so tests can redirect with cwd)
    scenario["output_dir"] = f"outputs/{scenario['name']}"

    dom = scenario.get("domain")

    def _fold_top_level_into(dst: Dict[str, Any], keys: Sequence[str]) -> None:
        """
        If raw has a top-level section (e.g., raw["horizon"]) and dst doesn't already,
        copy it into dst. Never overwrite explicit namespaced keys.
        """
        for k in keys:
            v = raw.get(k)
            if isinstance(v, dict) and k not in dst:
                dst[k] = dict(v)

    def _derive_horizon(dst: Dict[str, Any]) -> None:
        """
        If dst["horizon"] exists but missing n_steps/dt_hours, derive from run.horizon_hours and run.timestep_minutes.
        Does nothing if horizon is not a dict.
        """
        h = dst.setdefault("horizon", {})
        if not isinstance(h, dict):
            dst["horizon"] = {}
            h = dst["horizon"]

        # Only derive if missing/None (never overwrite explicit YAML)
        if h.get("n_steps") is None:
            try:
                horizon_hours = float(run["horizon_hours"])
                timestep_minutes = float(run["timestep_minutes"])
                if timestep_minutes > 0:
                    h["n_steps"] = int(round(horizon_hours * 60.0 / timestep_minutes))
            except Exception:
                pass

        if h.get("dt_hours") is None:
            try:
                timestep_minutes = float(run["timestep_minutes"])
                if timestep_minutes > 0:
                    h["dt_hours"] = float(timestep_minutes) / 60.0
            except Exception:
                pass

    # ------------------------------------------------------------
    # Phase 2: fold non-namespaced domain payloads based on domain
    # ------------------------------------------------------------
    if dom == "data_center":
        _fold_top_level_into(data_center, ["horizon", "series", "thermal", "cooling"])
        _derive_horizon(data_center)

    elif dom == "microgrid":
        _normalize_microgrid(microgrid, run)
        # Your microgrid YAML is domain-shaped: horizon/series/battery/grid at top-level.
        _fold_top_level_into(microgrid, ["horizon", "series", "battery", "grid", "penalties", "options"])
        # Optional: derive horizon if user omitted it
        _derive_horizon(microgrid)
        # Helpful: preserve name inside domain payload (mission uses cfg.get("name"))
        microgrid.setdefault("name", scenario["name"])

    elif dom == "water_network":
        _fold_top_level_into(water_network, ["horizon", "series", "tank", "pump"])
        _derive_horizon(water_network)
        water_network.setdefault("name", scenario["name"])

    # Build canonical dict with stable insertion order
    normalized: Dict[str, Any] = {
        "scenario": scenario,
        "run": run,
        "planner": planner,
        "objectives": objectives,
        "microgrid": microgrid,
        "data_center": data_center,
        "water_network": water_network,
    }

    return normalized


def validate_config(cfg: Mapping[str, Any]) -> None:
    """
    Friendly validation.

    Phase 1: strict microgrid golden path
    Phase 2: allow additional domains, still strict per-domain
    """
    cfg = _as_dict(cfg, "cfg")

    # ✅ FIRST: unknown top-level keys with hints (do this ONCE)
    for k in cfg.keys():
        if k not in CANONICAL_TOP_LEVEL_KEYS:
            suggestion = _suggest_key(k, CANONICAL_TOP_LEVEL_KEYS)
            hint = f"did you mean '{suggestion}'?" if suggestion else None
            raise ConfigError(k, f"unknown top-level key '{k}'", hint)

    scenario = _as_dict(cfg.get("scenario"), "scenario")
    run = _as_dict(cfg.get("run"), "run")
    planner = _as_dict(cfg.get("planner"), "planner")

    # Required
    _require_str(scenario, "name", "scenario")

    SUPPORTED_DOMAINS = {"microgrid", "data_center", "water_network"}

    # Default to microgrid if not specified (backward compatible)
    dom = scenario.get("domain", "microgrid")
    scenario["domain"] = dom

    SUPPORTED_DOMAINS = {"microgrid", "data_center", "water_network"}

    dom = _domain_from_cfg(cfg) or "microgrid"
    if not isinstance(dom, str) or not dom:
        raise ConfigError("domain", "missing/invalid domain (expected string or {name: ...})")

    if dom not in SUPPORTED_DOMAINS:
        raise ConfigError("domain", f"unsupported domain: {dom}")

    # ensure scenario.domain is canonical
    scenario = _as_dict(cfg.get("scenario"), "scenario")
    scenario["domain"] = dom

    if not isinstance(dom, str) or not dom:
        raise ConfigError("domain","missing/invalid domain (expected string or {name: ...})")

    if dom not in SUPPORTED_DOMAINS:
        raise ConfigError("domain", f"unsupported domain: {dom}")

    # Type checks
    seed = scenario.get("seed")
    if not isinstance(seed, int):
        raise ConfigError("scenario.seed", f"expected int, got {type(seed).__name__}")

    iters = planner.get("iterations")
    if not isinstance(iters, int) or iters < 0:
        raise ConfigError("planner.iterations", "must be a non-negative integer")

    # -------------------------
    # Domain-specific validation
    # -------------------------
    if dom == "microgrid":
        horizon = run.get("horizon_hours")
        if not isinstance(horizon, (int, float)) or float(horizon) <= 0:
            raise ConfigError("run.horizon_hours", "must be a positive number")

        ts = run.get("timestep_minutes")
        if not isinstance(ts, (int, float)) or float(ts) <= 0:
            raise ConfigError("run.timestep_minutes", "must be a positive number")

        # (Optional) ensure microgrid payload exists
        # _as_dict(cfg.get("microgrid"), "microgrid")

    elif dom == "data_center":
        dc = _as_dict(cfg.get("data_center"), "data_center")

        # Accept either data_center.horizon or (fallback) top-level horizon
        dc_horizon = dc.get("horizon")
        if isinstance(dc_horizon, dict):
            horizon_obj = dc_horizon
            horizon_path = "data_center.horizon"
        else:
            horizon_obj = cfg.get("horizon")
            horizon_path = "horizon"

        horizon = _as_dict(horizon_obj, horizon_path)

        n_steps = horizon.get("n_steps")
        if n_steps is None:
            raise ConfigError(f"{horizon_path}.n_steps", "must be a positive integer (got None)")

        # Normalize n_steps robustly (accept "24" or 24.0)
        try:
            n_steps_i = int(n_steps)
        except Exception:
            raise ConfigError(f"{horizon_path}.n_steps", f"must be a positive integer (got {n_steps!r})")

        if n_steps_i <= 0:
            raise ConfigError(f"{horizon_path}.n_steps", f"must be a positive integer (got {n_steps_i})")

        dt_hours = horizon.get("dt_hours")
        if not isinstance(dt_hours, (int, float)) or float(dt_hours) <= 0:
            raise ConfigError(f"{horizon_path}.dt_hours", f"must be a positive number (got {dt_hours!r})")
    elif dom == "water_network":
        wn = _as_dict(cfg.get("water_network"), "water_network")

        # Validate horizon
        horizon = _as_dict(wn.get("horizon"), "water_network.horizon")

        n_steps = horizon.get("n_steps")
        if n_steps is None:
            raise ConfigError("water_network.horizon.n_steps", "must be a positive integer (got None)")

        try:
            n_steps_i = int(n_steps)
        except Exception:
            raise ConfigError(
                "water_network.horizon.n_steps",
                f"must be a positive integer (got {n_steps!r})",
            )

        if n_steps_i <= 0:
            raise ConfigError(
                "water_network.horizon.n_steps",
                f"must be a positive integer (got {n_steps_i})",
            )

        dt_hours = horizon.get("dt_hours")
        if not isinstance(dt_hours, (int, float)) or float(dt_hours) <= 0:
            raise ConfigError(
                "water_network.horizon.dt_hours",
                f"must be a positive number (got {dt_hours!r})",
            )
# gaiaoptics/core/config.py
from __future__ import annotations

from dataclasses import dataclass
from os import error
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
import difflib

import yaml

from gaiaoptics.core.errors import ConfigError


CANONICAL_TOP_LEVEL_KEYS = ("scenario", "run", "planner", "objectives", "microgrid", "data_center", "water_network")


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


def normalize_config(raw: Mapping[str, Any], *, source_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Accepts "messy but acceptable" YAML and returns a canonical dict:
      - scenario/run/planner/objectives/<domain>
      - defaults filled
      - derived fields computed (scenario.name, scenario.output_dir)
    """
    raw = _as_dict(raw, "yaml")

    # --- Gather legacy shortcuts ---
    scenario_name = raw.get("scenario_name") or raw.get("name")
    domain = raw.get("domain") or raw.get("scenario", {}).get("domain")

    scenario = _as_dict(raw.get("scenario"), "scenario")
    run = _as_dict(raw.get("run"), "run")
    planner = _as_dict(raw.get("planner"), "planner")
    objectives = _as_dict(raw.get("objectives"), "objectives")

    # domain payloads
    microgrid = _as_dict(raw.get("microgrid"), "microgrid")
    data_center = _as_dict(raw.get("data_center"), "data_center")

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

    # ------------------------------------------------------------
    # Phase 2: Data center canonicalization (preserve nested keys!)
    # ------------------------------------------------------------
    dom = scenario.get("domain", None)

    # If user used top-level horizon/series/thermal/cooling (non-namespaced),
    # fold them into data_center ONLY when domain is data_center.
    if dom == "data_center":
        # Pull possible top-level sections
        top_horizon = raw.get("horizon")
        top_series = raw.get("series")
        top_thermal = raw.get("thermal")
        top_cooling = raw.get("cooling")

        if isinstance(top_horizon, dict) and "horizon" not in data_center:
            data_center["horizon"] = dict(top_horizon)
        if isinstance(top_series, dict) and "series" not in data_center:
            data_center["series"] = dict(top_series)
        if isinstance(top_thermal, dict) and "thermal" not in data_center:
            data_center["thermal"] = dict(top_thermal)
        if isinstance(top_cooling, dict) and "cooling" not in data_center:
            data_center["cooling"] = dict(top_cooling)

        # If data_center.horizon is missing pieces, derive from run (optional but helpful)
        h = data_center.setdefault("horizon", {})
        if not isinstance(h, dict):
            h = {}
            data_center["horizon"] = h

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

    # Build canonical dict with stable insertion order
    normalized: Dict[str, Any] = {
        "scenario": scenario,
        "run": run,
        "planner": planner,
        "objectives": objectives,
        "microgrid": microgrid,
        "data_center": data_center,
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

    if dom not in SUPPORTED_DOMAINS:
        raise ConfigError(
            "scenario.domain",
            f"unsupported domain '{dom}'. Supported domains: {sorted(SUPPORTED_DOMAINS)}",
        )

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
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from gaiaoptics.core.config import normalize_config, validate_config
from gaiaoptics.core.errors import ConfigError


def test_A_normalize_config_messy_defaults_and_derived(tmp_path: Path) -> None:
    messy = {
        "scenario_name": "microgrid_dispatch_demo",
        "seed": 7,
        "horizon_hours": 24,
        "timestep_minutes": 60,
        # legacy domain shortcut
        "domain": "microgrid",
        # domain payload present
        "microgrid": {"loads": [{"kw": 10.0}]},
        # partial planner
        "planner": {"iterations": 10},
        # typo key to ensure we don't fail here (we validate separately)
    }

    normalized = normalize_config(messy, source_path=tmp_path / "x.yaml")

    assert normalized["scenario"]["name"] == "microgrid_dispatch_demo"
    assert normalized["scenario"]["seed"] == 7
    assert normalized["scenario"]["output_dir"] == "outputs/microgrid_dispatch_demo"

    # defaults filled
    assert normalized["planner"]["name"] == "baseline_random"
    assert normalized["planner"]["restarts"] == 1
    assert normalized["objectives"]["cost_weight"] == 1.0
    assert normalized["objectives"]["emissions_weight"] == 1.0

    # stable canonical sections exist
    for k in ["scenario", "run", "planner", "objectives", "microgrid"]:
        assert k in normalized


def test_A_bad_yaml_raises_ConfigError_with_path_message_hint() -> None:
    bad = {
        "scneario": {"name": "oops"},  # typo top-level key
        "scenario": {"name": "x", "domain": "microgrid", "seed": "not-an-int"},
        "run": {"horizon_hours": -1},
        "planner": {"iterations": -5},
    }

    # First unknown key should trip with hint
    try:
        validate_config(bad)
        assert False, "expected ConfigError"
    except ConfigError as e:
        assert e.path == "scneario"
        assert "unknown top-level key" in e.message
        assert e.hint is not None and "did you mean" in e.hint


def test_B_output_contract_and_config_is_normalized(tmp_path: Path) -> None:
    # Create a messy YAML file
    yml = tmp_path / "demo.yaml"
    yml.write_text(
        """
scenario_name: microgrid_dispatch_demo
seed: 0
domain: microgrid
planner:
  iterations: 5
microgrid:
  loads:
    - kw: 10
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # Run CLI with outputs redirected to tmp_path/outputs
    outputs_root = tmp_path / "outputs"
    cmd = [
        sys.executable,
        "-m",
        "gaiaoptics",
        str(yml),
        "--outputs-root",
        str(outputs_root),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr + cp.stdout

    out_dir = outputs_root / "microgrid_dispatch_demo"
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "solution.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "traces.csv").exists()

    # config.yaml should be the *normalized* config, not raw
    cfg_written = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_written["planner"]["name"] == "baseline_random"  # default filled by normalize_config
    assert cfg_written["scenario"]["output_dir"] == "outputs/microgrid_dispatch_demo"


def test_C_microgrid_golden_path_deterministic(tmp_path: Path) -> None:
    yml = tmp_path / "demo.yaml"
    yml.write_text(
        """
scenario:
  name: microgrid_dispatch_demo
  domain: microgrid
  seed: 123
run:
  horizon_hours: 24
  timestep_minutes: 60
planner:
  name: baseline_random
  iterations: 5
microgrid:
  loads:
    - kw: 10
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def run_once(tag: str) -> Path:
        outputs_root = tmp_path / f"outputs_{tag}"
        cmd = [
            sys.executable, "-m", "gaiaoptics", str(yml),
            "--outputs-root", str(outputs_root),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        assert cp.returncode == 0, cp.stderr + cp.stdout
        return outputs_root / "microgrid_dispatch_demo"

    out1 = run_once("a")
    out2 = run_once("b")

    # Compare key metrics
    s1 = json.loads((out1 / "solution.json").read_text(encoding="utf-8"))
    s2 = json.loads((out2 / "solution.json").read_text(encoding="utf-8"))
    for k in ["feasible", "total_cost", "total_emissions", "worst_hard_margin"]:
        assert s1[k] == s2[k]

    # report headline section identical
    r1 = (out1 / "report.md").read_text(encoding="utf-8").splitlines()[:10]
    r2 = (out2 / "report.md").read_text(encoding="utf-8").splitlines()[:10]
    assert r1 == r2

    # traces.csv row count + totals match
    t1 = (out1 / "traces.csv").read_text(encoding="utf-8").strip().splitlines()
    t2 = (out2 / "traces.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(t1) == len(t2)
    assert t1[0] == t2[0]  # header

    # (optional) cheap total check by string equality if your traces are deterministic
    assert "\n".join(t1) == "\n".join(t2)
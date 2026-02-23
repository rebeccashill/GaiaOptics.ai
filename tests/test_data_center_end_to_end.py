# tests/test_data_center_end_to_end.py
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import yaml


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def _run_cli(yml_path: Path, outputs_root: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "gaiaoptics",
        str(yml_path),
        "--outputs-root",
        str(outputs_root),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0, (cp.stderr or "") + (cp.stdout or "")
    return outputs_root


def test_data_center_output_contract_smoke(tmp_path: Path) -> None:
    """
    End-to-end smoke test for the Phase 2 'data_center' domain.

    What this test enforces:
    - CLI accepts domain=data_center
    - output contract files exist
    - report.md is domain-correct (not microgrid text)
    """
    yml = tmp_path / "dc.yaml"
    _write_yaml(
        yml,
        """
        scenario:
          name: data_center_thermal_demo
          domain: data_center
          seed: 0

        # Keep domain payload under the domain key to match Phase 1 schema style:
        # scenario/run/planner/objectives/<domain>
        data_center:
          horizon:
            n_steps: 24
            dt_hours: 1
          series:
            it_load_kw: 50
            ambient_temp_c: 30
            price_per_kwh: 0.2
            carbon_kg_per_kwh: 0.4
          thermal:
            temp0_c: 24
            temp_max_c: 28
            thermal_mass_kwh_per_c: 50
            ua_kw_per_c: 2
            cop: 3
          cooling:
            p_max_kw: 30

        planner:
          name: baseline_random
          iterations: 5
        """,
    )

    outputs_root = tmp_path / "outputs"
    _run_cli(yml, outputs_root)

    out_dir = outputs_root / "data_center_thermal_demo"
    assert out_dir.exists()

    # Output contract
    for fname in ["config.yaml", "solution.json", "report.md", "traces.csv"]:
        assert (out_dir / fname).exists(), f"missing {fname}"

    # config.yaml should be normalized (at minimum, scenario.output_dir should exist)
    cfg_written = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_written["scenario"]["name"] == "data_center_thermal_demo"
    assert cfg_written["scenario"]["domain"] == "data_center"

    # solution.json should have Phase 1 compatibility keys (keeps summary/report tooling stable)
    sol = json.loads((out_dir / "solution.json").read_text(encoding="utf-8"))
    for k in ["feasible", "total_cost", "total_emissions", "worst_hard_margin"]:
        assert k in sol, f"solution.json missing key '{k}'"

    # Report must not be microgrid-labeled
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Microgrid Dispatch Report" not in report, "report.md still looks microgrid-specific"
    # And should mention the domain
    assert "data_center" in report, "report.md should mention domain=data_center"


def test_data_center_deterministic_two_runs(tmp_path: Path) -> None:
    """
    Runs the same data_center scenario twice and checks deterministic artifacts.
    """
    yml = tmp_path / "dc.yaml"
    _write_yaml(
        yml,
        """
        scenario:
          name: data_center_thermal_demo
          domain: data_center
          seed: 123

        data_center:
          horizon:
            n_steps: 24
            dt_hours: 1
          series:
            it_load_kw: 50
            ambient_temp_c: 30
            price_per_kwh: 0.2
            carbon_kg_per_kwh: 0.4
          thermal:
            temp0_c: 24
            temp_max_c: 28
            thermal_mass_kwh_per_c: 50
            ua_kw_per_c: 2
            cop: 3
          cooling:
            p_max_kw: 30

        planner:
          name: baseline_random
          iterations: 5
        """,
    )

    def run_once(tag: str) -> Path:
        outputs_root = tmp_path / f"outputs_{tag}"
        _run_cli(yml, outputs_root)
        return outputs_root / "data_center_thermal_demo"

    out1 = run_once("a")
    out2 = run_once("b")

    s1 = json.loads((out1 / "solution.json").read_text(encoding="utf-8"))
    s2 = json.loads((out2 / "solution.json").read_text(encoding="utf-8"))

    # If you keep these stable across domains, they’re a great contract to enforce.
    for k in ["feasible", "total_cost", "total_emissions", "worst_hard_margin"]:
        assert s1.get(k) == s2.get(k), f"solution.json differs on key '{k}'"

    # report header section identical (first ~12 lines)
    r1 = (out1 / "report.md").read_text(encoding="utf-8").splitlines()[:12]
    r2 = (out2 / "report.md").read_text(encoding="utf-8").splitlines()[:12]
    assert r1 == r2, "report.md header differs between runs"

    # traces.csv deterministic: header + row count must match
    t1 = (out1 / "traces.csv").read_text(encoding="utf-8").strip().splitlines()
    t2 = (out2 / "traces.csv").read_text(encoding="utf-8").strip().splitlines()
    assert t1 and t2
    assert t1[0] == t2[0], "traces.csv header differs"
    assert len(t1) == len(t2), "traces.csv row count differs"

    # Optional strict equality (enable if your traces are fully deterministic)
    assert "\n".join(t1) == "\n".join(t2), "traces.csv content differs"
# tests/test_warehouse_fleet_end_to_end.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from gaiaoptics.cli import main as cli_main


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.end_to_end
def test_warehouse_fleet_end_to_end(tmp_path: Path) -> None:
    """
    End-to-end smoke test for the warehouse_fleet domain:
    - runs the warehouse_fleet demo YAML via CLI
    - asserts output artifacts exist
    - asserts the solution is feasible
    """
    repo_root = Path(__file__).resolve().parents[1]
    scenario = repo_root / "examples" / "warehouse_fleet_demo.yaml"
    assert scenario.exists(), f"Missing scenario YAML: {scenario}"

    out_root = tmp_path / "outputs"

    rc = cli_main([str(scenario), "--out", str(out_root)])
    assert rc == 0, f"CLI returned non-zero exit code: {rc}"

    scenario_name = "warehouse_fleet_demo"
    out_dir = out_root / scenario_name
    assert out_dir.exists(), f"Missing output directory: {out_dir}"

    for fname in ["solution.json", "report.md", "config.yaml", "traces.csv"]:
        assert (out_dir / fname).exists(), f"missing {fname}"

    sol = _read_json(out_dir / "solution.json")

    # Output contract keys
    for k in ["feasible", "total_cost", "total_emissions", "worst_hard_margin"]:
        assert k in sol, f"solution.json missing key '{k}'"

    assert "constraints" in sol, "solution.json missing constraints"
    assert isinstance(sol["constraints"], list), "constraints should be a list"

    # The demo should come out feasible
    assert sol["feasible"] is True, (
        f"warehouse_fleet_demo should be feasible; "
        f"worst_hard_margin={sol.get('worst_hard_margin')}, "
        f"constraints={sol.get('constraints')}"
    )

    # Verify at least one HARD constraint is reported
    hard = [c for c in sol["constraints"] if isinstance(c, dict) and c.get("severity") == "HARD"]
    assert len(hard) > 0, "Expected at least one HARD constraint"

    # Report should mention the domain
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "warehouse" in report.lower(), (
        "report.md should mention warehouse"
    )

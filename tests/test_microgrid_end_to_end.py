# tests/test_microgrid_end_to_end.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from gaiaoptics.cli import main as cli_main


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.end_to_end
def test_microgrid_end_to_end(tmp_path: Path) -> None:
    """
    End-to-end smoke test:
    - runs the microgrid demo YAML via CLI
    - asserts artifacts exist
    - asserts solution contains constraints + traces (and is preferably feasible)
    """
    repo_root = Path(__file__).resolve().parents[1]
    scenario = repo_root / "examples" / "microgrid_dispatch_demo.yaml"
    assert scenario.exists(), f"Missing scenario YAML: {scenario}"

    out_root = tmp_path / "outputs"

    rc = cli_main(
        [
            str(scenario),
            "--out",
            str(out_root),
            "--iterations",
            "50",   # keep test fast
            "--restarts",
            "1",
            "--seed",
            "0",
            "--stop-on-feasible",
        ]
    )
    assert rc == 0, f"CLI returned non-zero exit code: {rc}"

    # Determine output dir (CLI uses scenario 'name' or stem, sanitized)
    scenario_name = "microgrid_dispatch_demo"
    out_dir = out_root / scenario_name
    assert out_dir.exists(), f"Missing output directory: {out_dir}"

    solution_path = out_dir / "solution.json"
    report_path = out_dir / "report.md"
    config_path = out_dir / "config.yaml"
    traces_path = out_dir / "traces.csv"

    assert solution_path.exists(), "solution.json was not written"
    assert report_path.exists(), "report.md was not written"
    assert config_path.exists(), "config.yaml was not written"
    # traces.csv may be absent if no series matched horizon, but in microgrid it should exist
    assert traces_path.exists(), "traces.csv was not written"

    sol = _read_json(solution_path)

    # Basic structure checks
    assert "feasibility" in sol, "solution.json missing feasibility"
    assert "constraints" in sol, "solution.json missing constraints"
    assert "traces" in sol, "solution.json missing traces"
    assert isinstance(sol["constraints"], list), "constraints should be a list"
    assert isinstance(sol["traces"], dict), "traces should be a dict"

    # Prefer feasible; allow non-feasible as long as constraints/traces exist
    feasible = bool(sol["feasibility"].get("feasible", False))
    if not feasible:
        # Still require meaningful diagnostics
        assert len(sol["constraints"]) > 0, "Non-feasible run should still report constraints"
        assert len(sol["traces"]) > 0, "Non-feasible run should still include traces"
    else:
        # If feasible, ensure at least one HARD constraint exists (SOC bounds, etc.)
        hard_constraints = [
            c for c in sol["constraints"] if isinstance(c, dict) and c.get("severity") == "HARD"
        ]
        assert len(hard_constraints) > 0, "Expected at least one HARD constraint for microgrid"
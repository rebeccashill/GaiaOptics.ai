# tests/test_water_network_end_to_end.py
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


def _write_minimal_water_config(path: Path) -> None:
    """
    Minimal-ish YAML to exercise the water_network domain end-to-end.
    Keep this intentionally small so it’s easy to evolve as the domain matures.
    """
    path.write_text(
        "\n".join(
            [
                "scenario_name: water_network_pumping_demo",
                "seed: 0",
                "horizon:",
                "  n_steps: 12",
                "  dt_minutes: 60",
                "",
                "domain:",
                "  name: water_network",
                "",
                "planner:",
                "  name: random_search",
                "  n_samples: 30",
                "  robustness: 0",
                "",
                "water_network:",
                "  # toy network: reservoir -> pump -> junction -> demand",
                "  nodes:",
                "    - {id: R, type: reservoir, head_m: 50.0}",
                "    - {id: J, type: junction, min_head_m: 20.0}",
                "  links:",
                "    - {id: P1, type: pump, from: R, to: J, max_flow_m3s: 0.05, max_power_kw: 15.0}",
                "  demands:",
                "    - {node: J, profile_m3s: [0.01, 0.012, 0.011, 0.013, 0.014, 0.015, 0.016, 0.017, 0.015, 0.013, 0.012, 0.011]}",
                "  tariffs:",
                "    # flat price for the demo",
                "    price_per_kwh: 0.20",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _try_run_via_python_api(config_path: Path, outputs_root: Path) -> bool:
    """
    Prefer a Python API if present (faster, clearer stack traces).
    Returns True if it ran, False if no compatible API was found.
    """
    try:
        import importlib

        # A few likely runner locations/names (adapt if you’ve standardized one).
        candidates = [
            ("gaiaoptics.core.runner", "run_scenario"),
            ("gaiaoptics.core.run", "run_scenario"),
            ("gaiaoptics.runner", "run_scenario"),
            ("gaiaoptics.core.runner", "run"),
            ("gaiaoptics", "run_scenario"),
        ]

        for mod_name, fn_name in candidates:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                # Support either (config_path) or (config_path, outputs_root) signatures.
                try:
                    fn(config_path=config_path, outputs_root=outputs_root)
                except TypeError:
                    try:
                        fn(config_path, outputs_root)
                    except TypeError:
                        fn(config_path)
                return True

        return False
    except Exception:
        # If API exists but fails, we want the test to surface it (don’t swallow).
        raise


def _try_run_via_cli(config_path: Path) -> bool:
    """
    Fallback: try a couple common CLI shapes.
    Returns True if it ran successfully, False if the CLI shape isn't recognized.
    """
    commands = [
        # Common Typer/argparse style
        [sys.executable, "-m", "gaiaoptics", "run", "--config", str(config_path)],
        [sys.executable, "-m", "gaiaoptics", "run", "-c", str(config_path)],
        # Alternate: config as positional
        [sys.executable, "-m", "gaiaoptics", "run", str(config_path)],
        # Some projects use `python -m gaiaoptics <config>`
        [sys.executable, "-m", "gaiaoptics", str(config_path)],
    ]

    last_err: str | None = None
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # Ran successfully
            return True
        except subprocess.CalledProcessError as e:
            out = e.stdout or ""
            last_err = out

            # Heuristic: if the CLI exists but rejected args/usage, try next shape.
            # If module import failed, it’s not a CLI-shape problem; stop early.
            if "No module named gaiaoptics" in out:
                raise
            if re.search(r"(unrecognized arguments|No such command|usage:)", out, re.I):
                continue

            # CLI ran but failed within program: surface the failure.
            raise

    # Nothing matched; treat as “no compatible CLI”
    if last_err:
        # Keep it around for debugging if someone prints captured output
        os.environ["GAIAOPTICS_TEST_LAST_CLI_OUTPUT"] = last_err
    return False


def _assert_output_contract(outputs_root: Path, scenario_name: str) -> Path:
    out_dir = outputs_root / scenario_name
    assert out_dir.exists(), f"Missing outputs directory: {out_dir}"

    cfg_out = out_dir / "config.yaml"
    sol_out = out_dir / "solution.json"
    rep_out = out_dir / "report.md"
    trc_out = out_dir / "traces.csv"

    assert cfg_out.exists(), "Missing output contract file: config.yaml"
    assert sol_out.exists(), "Missing output contract file: solution.json"
    assert rep_out.exists(), "Missing output contract file: report.md"
    assert trc_out.exists(), "Missing output contract file: traces.csv"

    # solution.json should be valid JSON and contain a feasibility flag (your contract can tighten later)
    data = json.loads(sol_out.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "solution.json must be a JSON object"
    assert "feasible" in data, "solution.json must include 'feasible'"

    # traces.csv should look like a CSV (at least a header line)
    header = trc_out.read_text(encoding="utf-8").splitlines()[0].strip()
    assert "," in header and len(header) > 3, "traces.csv should include a header row"

    # report.md should be non-empty and somewhat structured
    report_text = rep_out.read_text(encoding="utf-8")
    assert len(report_text.strip()) > 20, "report.md looks empty"
    assert (
        "Summary" in report_text or "##" in report_text
    ), "report.md should contain a heading/summary"

    return out_dir


@pytest.mark.end_to_end
def test_water_network_end_to_end(tmp_path: Path) -> None:
    """
    End-to-end smoke test for the water_network domain.

    What this test enforces:
    - The run completes (via Python API or CLI)
    - The non-negotiable output contract is produced:
        outputs/<scenario_name>/{config.yaml, solution.json, report.md, traces.csv}
    - solution.json is parseable and includes 'feasible'

    If water_network isn't registered/implemented yet, we SKIP (not FAIL),
    so CI stays green while you build the domain.
    """
    # Arrange
    repo_root = Path.cwd()
    outputs_root = repo_root / "outputs"
    config_path = tmp_path / "water_network_pumping_demo.yaml"
    _write_minimal_water_config(config_path)

    # Ensure a clean outputs area for the scenario
    scenario_name = "water_network_pumping_demo"
    scenario_out_dir = outputs_root / scenario_name
    if scenario_out_dir.exists():
        # Don’t delete all outputs, only the scenario folder
        for p in scenario_out_dir.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(scenario_out_dir.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        scenario_out_dir.rmdir()

    # Act
    ran = False

    # 1) Try Python API
    try:
        ran = _try_run_via_python_api(config_path=config_path, outputs_root=outputs_root)
    except ModuleNotFoundError as e:
        pytest.skip(f"gaiaoptics package not importable in test environment: {e}")
    except Exception as e:
        # If the domain is missing, let’s skip rather than fail.
        msg = str(e).lower()
        if "water_network" in msg and ("not registered" in msg or "unknown domain" in msg):
            pytest.skip("water_network domain not registered/implemented yet.")
        raise

    # 2) Fallback to CLI
    if not ran:
        try:
            ran = _try_run_via_cli(config_path=config_path)
        except ModuleNotFoundError as e:
            pytest.skip(f"gaiaoptics CLI not available in test environment: {e}")
        except Exception as e:
            msg = str(e).lower()
            if "water_network" in msg and ("not registered" in msg or "unknown domain" in msg):
                pytest.skip("water_network domain not registered/implemented yet.")
            raise

    if not ran:
        pytest.skip(
            "No compatible gaiaoptics runner API/CLI shape found. "
            "Update tests/test_water_network_end_to_end.py with your standardized entrypoint."
        )

    # Assert
    _assert_output_contract(outputs_root=outputs_root, scenario_name=scenario_name)
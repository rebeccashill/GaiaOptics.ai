# gaiaoptics/core/output.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import json

from gaiaoptics.core.config import canonical_yaml_dump


@dataclass
class RunArtifacts:
    normalized_config: Dict[str, Any]
    solution: Dict[str, Any]
    report_md: str
    traces_rows: List[Dict[str, Any]]  # list-of-dicts is simple + deterministic


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Deterministic column order: sorted keys across all rows
    cols = sorted({k for r in rows for k in r.keys()})
    lines = [",".join(cols)]
    for r in rows:
        parts = []
        for c in cols:
            v = r.get(c, "")
            # deterministic string conversion
            if v is None:
                s = ""
            elif isinstance(v, float):
                s = f"{v:.10g}"  # stable-ish
            else:
                s = str(v)
            # basic CSV escaping
            if any(ch in s for ch in [",", '"', "\n"]):
                s = '"' + s.replace('"', '""') + '"'
            parts.append(s)
        lines.append(",".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_output_contract(outputs_root: Path, scenario_name: str, artifacts: RunArtifacts) -> Path:
    """
    Always writes:
      outputs/<scenario_name>/
        config.yaml
        solution.json
        report.md
        traces.csv
    Returns the scenario output directory.
    """
    out_dir = outputs_root / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_text(out_dir / "config.yaml", canonical_yaml_dump(artifacts.normalized_config))
    _write_json(out_dir / "solution.json", artifacts.solution)
    _write_text(out_dir / "report.md", artifacts.report_md if artifacts.report_md.endswith("\n") else artifacts.report_md + "\n")
    _write_csv(out_dir / "traces.csv", artifacts.traces_rows)

    # At bottom of write_output_contract(...) after writing files
    try:
        from gaiaoptics.reporting.plots import generate_domain_plots

        domain = (artifacts.normalized_config.get("scenario", {}) or {}).get("domain", "unknown")
        generate_domain_plots(out_dir, domain=str(domain))
    except Exception:
        # Plots are optional polish; never fail the run for them.
        pass
    return out_dir
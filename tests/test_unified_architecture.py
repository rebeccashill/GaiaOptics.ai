# tests/test_unified_architecture.py
"""
Asserts that domains register via the same Problem interface (core untouched).

This test enforces the "unified kernel" contract:
- Each domain exposes build_problem_from_config(cfg) -> Problem
- The returned object is a gaiaoptics.core.problem.Problem
- The Problem has the standard hooks (simulate / constraints / objective)
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest

from gaiaoptics.core.problem import Problem


# Add/remove domains here as you create them.
DOMAIN_ENTRYPOINTS: List[str] = [
    "gaiaoptics.domains.microgrid.mission",
    "gaiaoptics.domains.data_center.mission",
    "gaiaoptics.domains.water_network.mission",
]


@pytest.mark.parametrize("module_path", DOMAIN_ENTRYPOINTS)
def test_domain_registers_via_problem_interface(module_path: str) -> None:
    """
    Domains must expose build_problem_from_config(cfg) and return a core Problem.
    """
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        pytest.skip(f"Domain not implemented yet: {module_path}")

    assert hasattr(mod, "build_problem_from_config"), (
        f"{module_path} must define build_problem_from_config(cfg) -> Problem"
    )

    builder = getattr(mod, "build_problem_from_config")
    assert callable(builder), f"{module_path}.build_problem_from_config must be callable"

    # Minimal cfg: domains should tolerate empty config (or provide defaults).
    cfg: Dict[str, Any] = {}
    problem = builder(cfg)

    assert isinstance(problem, Problem), (
        f"{module_path}.build_problem_from_config must return gaiaoptics.core.problem.Problem, "
        f"got {type(problem)!r}"
    )

    # Enforce the unified hook surface area.
    for attr in ("simulate", "constraints", "objective"):
        assert hasattr(problem, attr), f"Problem missing required hook: {attr}"
        fn = getattr(problem, attr)
        assert callable(fn), f"Problem hook must be callable: {attr}"

    # Optional but nice: ensure a stable name/id exists if your Problem supports it.
    # If you don't have these fields, delete this block.
    for opt_attr in ("name", "domain"):
        if hasattr(problem, opt_attr):
            val = getattr(problem, opt_attr)
            assert isinstance(val, str) and val.strip(), f"{opt_attr} must be a non-empty string"


def test_problem_type_is_from_core_only() -> None:
    """
    Sanity check: Problem type is defined in core and imported from there.
    This guards against domains redefining a shadow 'Problem' class.
    """
    assert Problem.__module__ == "gaiaoptics.core.problem", (
        f"Problem must come from gaiaoptics.core.problem (got {Problem.__module__})"
    )
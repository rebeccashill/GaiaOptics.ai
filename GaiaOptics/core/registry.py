# gaiaoptics/core/registry.py
"""
Domain registry for GaiaOptics.

This module maps a string domain name (from config.yaml)
to a builder function that returns a Problem.

Example usage:

    from gaiaoptics.core.registry import get_domain_builder

    builder = get_domain_builder("warehouse_fleet")
    problem = builder(config_dict)

Domain modules should call:

    register_domain("warehouse_fleet", build_problem)

at import time (usually inside gaiaoptics/domains/__init__.py).
"""

from __future__ import annotations

from typing import Callable, Dict, List

from gaiaoptics.core.problem import Problem


# Type alias for builder signature
DomainBuilder = Callable[[dict], Problem]


# Internal storage
_DOMAINS: Dict[str, DomainBuilder] = {}


# ----------------------------
# Public API
# ----------------------------

def register_domain(name: str, builder: DomainBuilder) -> None:
    """
    Register a domain builder.

    Parameters:
        name: domain string used in config.yaml
        builder: function that takes config dict and returns Problem
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Domain name must be a non-empty string")

    if not callable(builder):
        raise TypeError("Domain builder must be callable")

    if name in _DOMAINS:
        raise RuntimeError(f"Domain '{name}' already registered")

    _DOMAINS[name] = builder


def get_domain_builder(name: str) -> DomainBuilder:
    """
    Retrieve a registered domain builder.

    Raises KeyError if domain is not registered.
    """
    if name not in _DOMAINS:
        available = ", ".join(sorted(_DOMAINS.keys())) or "(none)"
        raise KeyError(f"Unknown domain '{name}'. Available domains: {available}")
    return _DOMAINS[name]


def list_domains() -> List[str]:
    """
    Return sorted list of registered domain names.
    """
    return sorted(_DOMAINS.keys())
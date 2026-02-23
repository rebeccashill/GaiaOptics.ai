# gaiaoptics/reporting/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Optional


def generate_domain_plots(outputs_dir: Path, domain: str) -> Optional[Path]:
    """
    Generate a single credibility plot per domain into outputs_dir/plots/.

    Returns the path to the plot if created, else None.
    Never raises unless outputs_dir is missing (callers may choose to catch anyway).
    """
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    # If matplotlib isn't installed, skip quietly (keeps core runnable).
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return None

    domain = (domain or "").strip()

    if domain == "microgrid":
        from gaiaoptics.visualization.microgrid_plots import plot_microgrid_outputs_dir

        return plot_microgrid_outputs_dir(outputs_dir)

    if domain == "data_center":
        # you'll create this file next (data_center_plots.py)
        from gaiaoptics.visualization import data_center_plots

        plot_fn = getattr(data_center_plots, "plot_data_center_outputs_dir", None)
        if plot_fn is None:
            return None
        return plot_fn(outputs_dir)

    if domain == "water_network":
        # you'll create this file next (water_network_plots.py)
        from gaiaoptics.visualization import water_network_plots

        plot_fn = getattr(water_network_plots, "plot_water_network_outputs_dir", None)
        if plot_fn is None:
            return None
        return plot_fn(outputs_dir)

    # Unknown domain => no plots
    return None
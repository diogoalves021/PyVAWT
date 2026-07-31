"""
Geometric and aerodynamic helper functions for VAWT simulations.

Provides utilities for Mach number estimation, airfoil thickness-to-chord ratio
extraction, stall angle detection via NeuralFoil, and turbine geometry resolution.
"""

from __future__ import annotations

from typing import Any
import numpy as np

# Try importing UI helper with fallback
try:
    from src.pyvawt.ui.ui import UI
except ImportError:
    class UI:  # type: ignore
        @staticmethod
        def status(category: str, message: str, level: str = "info") -> None:
            print(f"[{category}] {message}")


# ==============================================================================
# GEOMETRIC HELPERS
# ==============================================================================

def resolve_turbine_geometry(
    turbine_params: dict[str, Any], verbose: bool = True
) -> tuple[float, float, float]:
    """
    Synchronize physical geometry parameters for rotor consistency.

    Calculates missing parameters (radius or solidity) from blade count,
    chord length, and explicit target inputs.

    Parameters
    ----------
    turbine_params : dict
        Dictionary containing turbine geometric properties ('B', 'chord', 'solidity', 'r').
    verbose : bool, default=True
        Whether to log geometry updates.

    Returns
    -------
    r : float
        Rotor radius [m].
    chord : float
        Sanitized blade chord length [m].
    solidity : float
        Rotor solidity [-].
    """
    B = float(np.squeeze(turbine_params["B"]))
    chord = float(np.squeeze(turbine_params["chord"]))
    r_yaml = turbine_params.get("r")

    if "solidity" in turbine_params and turbine_params["solidity"] is not None:
        solidity = float(np.squeeze(turbine_params["solidity"]))
        r = float((B * chord) / solidity)
    else:
        if r_yaml is None:
            raise ValueError(
                "Configuration Error: Provide at least Radius ('r') or Solidity ('solidity')."
            )
        r = float(np.squeeze(r_yaml))
        solidity = float((B * chord) / r)

    turbine_params["r"] = r
    turbine_params["chord"] = chord
    turbine_params["solidity"] = solidity

    return r, chord, solidity


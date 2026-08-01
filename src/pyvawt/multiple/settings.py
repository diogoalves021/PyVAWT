"""
Configuration Management Module for PyVAWT.

Provides cached loading, validation, and parameter normalization of YAML
configuration files for Vertical Axis Wind Turbine (VAWT) simulations.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Type alias for scalar floats or NumPy arrays used across configuration and solvers
ArrayOrFloat = float | np.ndarray


@lru_cache(maxsize=1)
def load_config(path: str | Path = "src/pyvawt/config/config_multiple.yaml") -> dict[str, Any]:
    """
    Load, validate, and normalize a YAML simulation configuration file.

    The result is cached in memory via `lru_cache` to eliminate repeated
    disk I/O operations during iterative simulations or sweep routines.

    Parameters
    ----------
    path : str or pathlib.Path, default="src/pyvawt/config/config_multiple.yaml"
        Path to the target YAML configuration file.

    Returns
    -------
    dict of {str : Any}
        Parsed, validated, and normalized configuration dictionary containing
        at least the mandatory sections: 'turbine', 'environment', and 'solver'.

    Raises
    ------
    FileNotFoundError
        If no configuration file exists at the specified path.
    ValueError
        If the configuration file is empty or contains invalid YAML structure.
    KeyError
        If any mandatory top-level section ('turbine', 'environment', or 'solver')
        is missing from the configuration.

    Notes
    -----
    To maintain seamless multi-turbine support across solver modules, single scalar
    parameters (e.g., `chord`, `solidity`, `Vinf`, `airfoil`) are automatically
    normalized into Python lists during loading.
    """
    filepath = Path(path)

    # 1. Validate file existence
    if not filepath.is_file():
        raise FileNotFoundError(f"Configuration file not found: '{filepath.resolve()}'")

    # 2. Parse YAML file safely
    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Configuration file '{filepath}' is empty or invalid.")

    # 3. Ensure essential schema sections exist
    for section in ("turbine", "environment", "solver"):
        if section not in config:
            raise KeyError(f"Missing mandatory section '{section}' in configuration.")

    # 4. Helper function to enforce list structures for multi-turbine compatibility
    def _ensure_list(target_dict: dict[str, Any], key: str) -> None:
        """Wrap scalar dictionary entry in a list if it is not already one."""
        if key in target_dict and not isinstance(target_dict[key], list):
            target_dict[key] = [target_dict[key]]

    # Normalize scalar entries to lists
    _ensure_list(config["turbine"], "chord")
    _ensure_list(config["turbine"], "solidity")
    _ensure_list(config["environment"], "Vinf")

    nf_cfg = config.get("solver", {}).get("neuralfoil", {})
    _ensure_list(nf_cfg, "airfoil")

    return config


def _format_value(value: Any) -> str:
    """Format configuration primitive values into clean terminal strings."""
    if isinstance(value, bool):
        return "Enabled" if value else "Disabled"
    if isinstance(value, list) and len(value) == 1:
        return str(value[0])
    if isinstance(value, dict):
        formatted_items = [
            f"{k}={_format_value(v)}"
            for k, v in value.items()
            if not isinstance(v, dict)
        ]
        return ", ".join(formatted_items) if formatted_items else str(value)
    return str(value)


def display_multi_config(config: dict[str, Any]) -> None:
    """
    Print multi-turbine simulation configuration in standardized CLI style.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary containing layout, environment, solver,
        submodels, and output parameters.
    """
    print("\n─── SIMULATION CONFIGURATION ────────────────────────────────────\n")

    # 1. Turbine & Layout
    turbine_cfg = config.get("turbine", {})
    if turbine_cfg:
        print("  [TURBINE LAYOUT & ARRAY]")
        
        # Extract turbine position vectors
        center_x = turbine_cfg.get("centerX", [0.0])
        center_y = turbine_cfg.get("centerY", [0.0])
        
        if not isinstance(center_x, list):
            center_x = [center_x]
        if not isinstance(center_y, list):
            center_y = [center_y]

        num_turbines = max(len(center_x), len(center_y))
        print(f"  • {'Total Turbines':<30} : {num_turbines}")

        for i in range(num_turbines):
            x_pos = center_x[i] if i < len(center_x) else center_x[-1]
            y_pos = center_y[i] if i < len(center_y) else center_y[-1]
            sub_label = f"Turbine #{i+1}"
            print(f"    - {sub_label:<28} : Position=({x_pos}, {y_pos}) m")

        key_aliases_turb = {
            "r": "Radius (r)",
            "height": "Height",
            "twist": "Twist Angle",
            "delta": "Cone/Inclin. Angle (delta)",
            "chord": "Blade Chord",
            "B": "Number of Blades (B)",
            "solidity": "Solidity",
            "Omega": "Rotational Speed (Omega)",
            "ntheta": "Azimuthal Discretization",
        }
        for k, label in key_aliases_turb.items():
            if k in turbine_cfg:
                print(f"  • {label:<30} : {_format_value(turbine_cfg[k])}")
        print()

    # 2. Environment
    env_cfg = config.get("environment", {})
    if env_cfg:
        print("  [ENVIRONMENT & FLOW FIELD]")
        key_aliases_env = {
            "Vinf": "Freestream Velocity (Vinf)",
            "rho": "Air Density (rho)",
            "mu": "Dynamic Viscosity (mu)",
        }
        for k, v in env_cfg.items():
            label = key_aliases_env.get(k, k)
            print(f"  • {label:<30} : {_format_value(v)}")
        print()

    # 3. Solver
    solver_cfg = config.get("solver", {})
    if solver_cfg:
        print("  [SOLVER & INTERACTION]")
        key_aliases_solver = {
            "num_turbines": "Configured Turbines Count",
            "method": "Polar Generator Method",
            "fixed_parameter": "Fixed Parameter",
            "neuralfoil": "NeuralFoil Settings",
            "file": "Experimental File Source",
            "simulation3d": "3D Simulation Setup",
        }
        for k, v in solver_cfg.items():
            label = key_aliases_solver.get(k, k)
            print(f"  • {label:<30} : {_format_value(v)}")
        print()

    # 4. Submodels
    submodels_cfg = config.get("submodels", {})
    if submodels_cfg:
        print("  [SUBMODELS]")
        key_aliases_sub = {
            "tip_loss": "Tip Loss Correction",
            "dynamic_stall": "Dynamic Stall Model",
            "flow_curvature": "Flow Curvature Model",
        }
        for k, v in submodels_cfg.items():
            label = key_aliases_sub.get(k, k)
            print(f"  • {label:<30} : {_format_value(v)}")
        print()

    # 5. Output Settings
    output_cfg = config.get("output", {})
    if output_cfg:
        print("  [OUTPUT SETTINGS]")
        key_aliases_out = {
            "save": "Save Data Output",
            "save_config": "Save Config File",
            "save_plot": "Save Plots",
            "data_file": "Data Export Format",
            "plot_image": "Plot Resolution/Format",
        }
        for k, v in output_cfg.items():
            label = key_aliases_out.get(k, k)
            print(f"  • {label:<30} : {_format_value(v)}")
        print()

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the multi-turbine simulation module."""
    parser = argparse.ArgumentParser(
        description="PyVAWT - Multi-Turbine Aerodynamic Solver Module"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Display resolved simulation configuration.",
    )
    return parser.parse_args()

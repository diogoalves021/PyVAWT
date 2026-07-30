"""
Data Generation and Airfoil Aerodynamics Module.

Provides YAML configuration loading utilities and NeuralFoil-based aerodynamic
coefficient (Cl, Cd) evaluations for Vertical Axis Wind Turbine (VAWT) simulations.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import aerosandbox as asb
import numpy as np
import yaml

# Type alias supporting both single float values and NumPy arrays
ArrayOrFloat = Union[float, np.ndarray]


@lru_cache(maxsize=1)
def load_config(path: str | Path = "src/pyvawt/config/config_multiple.yaml") -> dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary, cached in memory to optimize performance.

    Parameters
    ----------
    path : str or Path, default="src/pyvawt/config/config_multiple.yaml"
        Path to the target YAML configuration file.

    Returns
    -------
    config : dict[str, Any]
        Parsed and validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the specified path.
    ValueError
        If the YAML file is empty or invalid.
    KeyError
        If mandatory configuration sections are missing.
    """
    filepath = Path(path)

    if not filepath.is_file():
        raise FileNotFoundError(f"Configuration file not found: '{filepath.resolve()}'")

    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file '{filepath}' is empty or invalid.")

    # Validate required top-level sections
    required_sections = ("turbine", "environment", "solver")
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Mandatory section '{section}' is missing from configuration file.")

    def _ensure_list(target_dict: dict[str, Any], key: str) -> None:
        """Enforces that a target dictionary entry is formatted as a list."""
        if key in target_dict and not isinstance(target_dict[key], list):
            target_dict[key] = [target_dict[key]]

    _ensure_list(config["turbine"], "chord")
    _ensure_list(config["turbine"], "solidity")
    _ensure_list(config["environment"], "Vinf")

    # Ensure 'airfoil' inside solver.neuralfoil is formatted as a list
    nf_cfg = config.get("solver", {}).get("neuralfoil", {})
    _ensure_list(nf_cfg, "airfoil")

    return config


@lru_cache(maxsize=16)
def _get_airfoil_instance(name: str) -> asb.Airfoil:
    """Cache AeroSandbox Airfoil instances to eliminate redundant initialization overhead."""
    return asb.Airfoil(name=name)


def get_cl_cd_neuralfoil(
    alpha: ArrayOrFloat,
    W: ArrayOrFloat,
    turbine_index: int = 0,
    airfoil_index: int = 0,
    config_path: str | Path = "src/pyvawt/config/config_multiple.yaml",
) -> tuple[ArrayOrFloat, ArrayOrFloat]:
    """
    Evaluate lift (Cl) and drag (Cd) coefficients using the NeuralFoil model via AeroSandbox.

    Parameters
    ----------
    alpha : float or np.ndarray
        Angle of attack in radians.
    W : float or np.ndarray
        Local apparent wind speed magnitude in m/s.
    turbine_index : int, default=0
        Index of the target turbine used to select chord length from configuration.
    airfoil_index : int, default=0
        Index of the target airfoil profile used to select airfoil name from configuration.
    config_path : str or Path, default="src/pyvawt/config/config_multiple.yaml"
        Path to the configuration file loaded via `load_config`.

    Returns
    -------
    cl : float or np.ndarray
        Interpolated lift coefficient(s) matching input shape.
    cd : float or np.ndarray
        Interpolated drag coefficient(s) matching input shape.
    """
    config = load_config(config_path)

    # Resilient parameters extraction for turbine and environment
    chords = config["turbine"]["chord"]
    chord = float(chords[turbine_index % len(chords)])

    rho = float(config["environment"]["rho"])
    mu = float(config["environment"]["mu"])

    # Access NeuralFoil sub-configuration dictionary
    solver_cfg = config.get("solver", {})
    nf_cfg = solver_cfg.get("neuralfoil", {})

    # Extract NeuralFoil airfoils list and execution parameters
    airfoils = nf_cfg.get("airfoil", ["naca0018"])
    airfoil_name = str(airfoils[airfoil_index % len(airfoils)])

    model_size = nf_cfg.get("model_size", "large")
    include_360 = nf_cfg.get("include_360_deg_effects", True)

    # Calculate dimensionless fluid mechanics parameters
    Re = rho * W * chord / mu
    mach = W / 343.2  # Speed of sound in air (~343.2 m/s)

    alpha_arr = np.asarray(alpha, dtype=np.float64)

    airfoil = _get_airfoil_instance(airfoil_name)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=np.rad2deg(alpha_arr),
        Re=Re,
        mach=mach,
        model_size=model_size,
        include_360_deg_effects=include_360,
    )

    cl_res = np.asarray(aero["CL"]).reshape(alpha_arr.shape)
    cd_res = np.asarray(aero["CD"]).reshape(alpha_arr.shape)

    # Return scalar float values if input was scalar
    if np.isscalar(alpha) and np.isscalar(W):
        return float(cl_res), float(cd_res)

    return cl_res, cd_res

"""
Configuration Management Module for PyVAWT.

Provides cached loading, validation, and parameter normalization of YAML
configuration files for Vertical Axis Wind Turbine (VAWT) simulations.
"""
from __future__ import annotations

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

"""
Utility and execution support functions for VAWT simulations.

Provides JIT engine warm-up routines, worker initialization for parallel execution,
YAML configuration loading/saving, and CLI argument parsing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Try importing UI helper with fallback
try:
    from src.pyvawt.ui.ui import UI
except ImportError:
    class UI:  # type: ignore
        @staticmethod
        def status(category: str, message: str, level: str = "info") -> None:
            print(f"[{category}] {message}")


# ==============================================================================
# JIT NUMERICS & KERNELS
# ==============================================================================

def warmup_numba_kernels(verbose: bool = True) -> None:
    """
    Pre-compile or load cached Numba JIT kernels prior to simulation execution.

    Parameters
    ----------
    verbose : bool, default=True
        If True, displays initialization status and compilation elapsed time.
    """
    if verbose:
        UI.status("JIT Engine (Numba)", "Compiling C kernels...", level="info")

    t0 = time.perf_counter()
    dummy_1d = np.zeros(10, dtype=np.float64)
    dummy_grid = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
    dummy_table = np.zeros((10, 10), dtype=np.float64)

    try:
        from src.pyvawt.single.simulation import _radialforce_kernel

        _radialforce_kernel(
            dummy_1d,
            dummy_1d,
            dummy_grid,
            10.0,
            1.0,
            0.0,
            0.0,
            3,
            10.0,
            0.1,
            10.0,
            1.2,
            1.8e-5,
            dummy_grid,
            dummy_grid,
            dummy_table,
            dummy_table,
            True,
            0.2,
            -0.2,
            0.0,
            0.12,
        )
        if verbose:
            dt = time.perf_counter() - t0
            UI.status("JIT Engine (Numba)", f"Ready ({dt:.2f}s)", level="ok")
    except Exception as e:
        if verbose:
            UI.status("JIT Engine (Numba)", f"Failed: {e}", level="warn")


def _worker_init() -> None:
    """
    Initializer function for parallel worker processes.

    Triggers silent compilation and loading of Numba kernels within each spawned
    multiprocessing process pool worker.
    """
    warmup_numba_kernels(verbose=False)


# ==============================================================================
# CONFIGURATION & FILE I/O
# ==============================================================================

def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load simulation configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the `.yaml` configuration file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing simulation parameters.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    yaml.YAMLError
        If parsing fails.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {path_obj}:\n{e}")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """
    Save configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        Dictionary with simulation parameters to save.
    path : str or Path
        Output path for the `.yaml` file.
    """
    path_obj = Path(path)
    if path_obj.suffix.lower() != ".yaml":
        path_obj = path_obj.with_suffix(".yaml")

    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)


def parse_args() -> argparse.Namespace:
    """
    Parse Command-Line Interface (CLI) arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run VAWT actuator-cylinder simulations."
    )

    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show full configuration before running simulation",
    )

    return parser.parse_args()

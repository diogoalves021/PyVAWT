"""
Results export, artifact management, and disk persistence helpers for VAWT simulations.

Provides functions for creating execution output directories, saving 2D sweep log
tables to CSV, exporting 3D power coefficient numerical data and plots, and reading
tabular .dat files.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.pyvawt.single.utils import save_config


# ==============================================================================
# DIRECTORY & FILE HELPERS
# ==============================================================================

# Default hardcoded root output directory
BASE_RESULTS_DIR = Path("results")


def create_run_directory(
    config: dict[str, Any],
    base_dir: str | Path = BASE_RESULTS_DIR,
) -> Path:
    """
    Create a unique timestamped run directory and save a configuration snapshot.

    Parameters
    ----------
    config : dict
        Full simulation configuration dictionary.
    base_dir : str or Path, default=BASE_RESULTS_DIR
        Base root directory where output runs will be stored.

    Returns
    -------
    Path
        Path object pointing to the newly created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sim3d = config.get("solver", {}).get("simulation3d", {})
    is_3d = sim3d.get("enabled", False)

    if is_3d:
        h = config.get("turbine", {}).get("height", 0.0)
        slices = sim3d.get("settings", {}).get("vertical_layers", 0)
        param_tag = f"3D_H{h}_Ns{slices}"
    else:
        sol = config.get("turbine", {}).get("solidity", [0.0])
        sol_val = sol[0] if isinstance(sol, list) else sol
        param_tag = f"2D_sol{sol_val}"

    run_dir = Path(base_dir) / f"{timestamp}_{param_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config_used.yaml")
    return run_dir

def setup_output_dir(base_path: str | Path, run_name: str) -> Path:
    """
    Create and return the output directory path.

    Parameters
    ----------
    base_path : str or Path
        Base directory path where execution folders are stored.
    run_name : str
        Name of the specific run folder to create.

    Returns
    -------
    Path
        Path object pointing to the created directory.
    """
    out_dir = Path(base_path) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_dat(path: str | Path) -> list[list[float]]:
    """
    Read a `.dat` tabular numerical file into a nested list of floats.

    Parameters
    ----------
    path : str or Path
        File path to read.

    Returns
    -------
    list[list[float]]
        Parsed rows of numerical values.
    """
    data: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header line
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            data.append([float(x) for x in stripped.split()])
    return data


# ==============================================================================
# EXPORT HELPERS
# ==============================================================================

def export_2d_results(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: str | Path = "results/2D",
) -> Path | None:
    """
    Export 2D sweep simulation results to a CSV log and copy configuration.

    Parameters
    ----------
    results : list of dict
        List of dictionaries containing output metrics for each simulation case.
    config : dict
        Full simulation configuration dictionary.
    output_dir : str or Path, default="results/2D"
        Base directory where 2D execution results will be stored.

    Returns
    -------
    Path or None
        Path to the generated CSV log file, or None if `results` is empty.
    """
    if not results:
        return None

    out_path = setup_output_dir(output_dir, "batch_execution")

    csv_file = out_path / "log_simulacoes.csv"
    fieldnames = list(results[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    save_config(config, out_path / "config_used.yaml")

    return csv_file


def export_3d_results(
    config: dict[str, Any],
    tsr: np.ndarray,
    cp_3d: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """
    Export integrated 3D numerical data, power coefficient plot, and config snapshot.

    Parameters
    ----------
    config : dict
        Full simulation configuration dictionary used in the run.
    tsr : np.ndarray
        Array of Tip Speed Ratio values.
    cp_3d : np.ndarray
        Array of integrated global 3D Power Coefficients.
    output_dir : str or Path
        Target execution run directory.

    Returns
    -------
    Path
        Directory Path object where output artifacts were saved.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save active configuration snapshot inside the target folder
    if config.get("output", {}).get("save_config", True):
        save_config(config, out_path / "config_used.yaml")

    # Extract plot configuration options
    plot_cfg = config.get("output", {}).get("plot_image", {})
    dpi = plot_cfg.get("dpi", 300)
    image_format = plot_cfg.get("format", "png")

    # Save numerical data table
    data_to_save = np.column_stack((tsr, cp_3d))
    np.savetxt(
        out_path / "results_3D.dat",
        data_to_save,
        header="TSR\tCp_3D",
        fmt="%.6f",
        delimiter="\t",
    )

    # Save power coefficient visualization plot
    plt.figure()
    plt.plot(tsr, cp_3d, "b-o", label="$C_p$ 3D")
    plt.xlabel("TSR")
    plt.ylabel("$C_p$ 3D")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / f"cp_curve_3D.{image_format}", dpi=dpi)
    plt.close()

    return out_path

"""
Results export, artifact management, and disk persistence helpers for VAWT simulations.

Provides functions for creating execution output directories, saving 2D sweep log
tables to CSV, exporting 3D power coefficient numerical data and plots, and reading
tabular .dat files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.pyvawt.single.utils import save_config


# ==============================================================================
# DIRECTORY & FILE HELPERS
# ==============================================================================

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
    tsr: np.ndarray,
    cp_3d: np.ndarray,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """
    Export 3D simulation numerical data, Cp curve plot, and configuration copy.

    Parameters
    ----------
    tsr : np.ndarray
        Array of Tip Speed Ratio values [-].
    cp_3d : np.ndarray
        Array of integrated 3D Power Coefficients [-].
    config : dict
        Full simulation configuration dictionary.
    output_dir : str or Path
        Target directory path for exporting 3D artifacts.

    Returns
    -------
    Path
        Directory Path object where artifacts were saved.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_to_save = np.column_stack((tsr, cp_3d))
    np.savetxt(
        out_path / "results_3D.dat",
        data_to_save,
        header="TSR\tCp_3D",
        fmt="%.6f",
        delimiter="\t",
    )

    plt.figure()
    plt.plot(tsr, cp_3d, "b-o", label="$C_p$ 3D")
    plt.xlabel("TSR")
    plt.ylabel("$C_p$ 3D")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / "cp_curve_3D.png", dpi=300)
    plt.close()

    save_config(config, out_path / "config_used.yaml")

    return out_path

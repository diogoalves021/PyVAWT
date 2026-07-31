"""
Data Export and Visualization Module for PyVAWT.

Provides utilities for exporting raw numerical simulation data (DAT/CSV),
saving YAML configuration snapshots, generating 2D turbine layout diagrams,
and rendering performance metric curves ($C_p$ vs. $\lambda$).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.pyvawt.multiple.simulation import Turbine

DEFAULT_RESULTS_DIR: str = "results"


def _apply_plot_style() -> None:
    """
    Configure global Matplotlib parameters for publication-quality figures.

    Adjusts default font family, base font sizes, axis label sizes, and legend
    formatting across matplotlib figures.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
    })


def plot_turbine_layout(
    turbines: list[Turbine], 
    case_dir: Path, 
    fmt: str = "png", 
    dpi: int = 300
) -> None:
    """
    Generate and save a 2D spatial layout diagram of the turbine array.

    Renders a top-down view showing turbine center coordinates and rotor swept
    diameters along the horizontal plane ($X, Y$).

    Parameters
    ----------
    turbines : list of Turbine
        List of initialized `Turbine` instances representing the array.
    case_dir : Path
        Directory path where the layout image will be saved.
    fmt : str, default="png"
        Output image file extension (e.g., `"png"`, `"pdf"`, `"svg"`).
    dpi : int, default=300
        Resolution in dots per inch for raster graphics formats.
    """
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, t in enumerate(turbines):
        ax.plot(t.centerX, t.centerY, "xr", markersize=10, label="Center" if i == 0 else "")
        circle = plt.Circle((t.centerX, t.centerY), t.r, color="blue", fill=False, linestyle="--", alpha=0.6)
        ax.add_patch(circle)
        ax.text(t.centerX, t.centerY + t.r * 1.1, f"Turbine {i+1}", ha="center", fontsize=12, weight="bold")

    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_aspect("equal", "box")
    fig.tight_layout()

    out_path = case_dir / f"layout.{fmt.lower()}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_raw_data(
    case_dir: Path,
    num_turbines: int,
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray,
    ct_vec: np.ndarray,
    rp_vec: np.ndarray,
    tp_vec: np.ndarray,
    zp_vec: np.ndarray,
    fmt: str = "dat",
    include_header: bool = True
) -> None:
    """
    Export raw simulation vectors to plain-text files for each turbine.

    Exports Tip Speed Ratio ($\lambda$), Power Coefficient ($C_p$), Thrust
    Coefficient ($C_t$), Radial Force ($R_p$), Tangential Force ($T_p$), and
    Axial Force ($Z_p$) per turbine into tab- or comma-delimited files.

    Parameters
    ----------
    case_dir : Path
        Target folder path for results storage.
    num_turbines : int
        Total number of turbines in the array.
    tsr_vec : np.ndarray, shape (N_tsr,)
        Array of Tip Speed Ratio ($\lambda$) points.
    cp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Array of Power Coefficients ($C_p$).
    ct_vec : np.ndarray, shape (N_tsr, N_turbines)
        Array of Thrust Coefficients ($C_t$).
    rp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Array of Radial Force Coefficients ($R_p$).
    tp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Array of Tangential Force Coefficients ($T_p$).
    zp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Array of Axial Force Coefficients ($Z_p$).
    fmt : str, default="dat"
        Target file format (`"dat"` for tab-delimited or `"csv"` for comma-delimited).
    include_header : bool, default=True
        Whether to write column header names in the output files.
    """
    is_csv = (fmt.lower() == "csv")
    delimiter = "," if is_csv else "\t"
    ext = "csv" if is_csv else "dat"

    header = f"TSR{delimiter}CP{delimiter}CT{delimiter}Rp{delimiter}Tp{delimiter}Zp" if include_header else ""

    for t in range(num_turbines):
        data_to_save = np.column_stack((
            tsr_vec,
            cp_vec[:, t],
            ct_vec[:, t],
            rp_vec[:, t],
            tp_vec[:, t],
            zp_vec[:, t]
        ))
        out_filename = case_dir / f"results_t{t+1}.{ext}"
        np.savetxt(out_filename, data_to_save, header=header, fmt="%.6f", delimiter=delimiter, comments="")


def _plot_performance_curves(
    case_dir: Path,
    num_turbines: int,
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray,
    fmt: str = "png",
    dpi: int = 300
) -> None:
    """
    Generate and save Power Coefficient ($C_p$) vs. Tip Speed Ratio ($\lambda$) curves.

    Plots individual turbine performance curves alongside the system average
    curve when multiple turbines are present in the simulation setup.

    Parameters
    ----------
    case_dir : Path
        Directory path where the plot image will be written.
    num_turbines : int
        Total number of turbines in the system.
    tsr_vec : np.ndarray, shape (N_tsr,)
        Tip Speed Ratio ($\lambda$) range array.
    cp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Matrix containing $C_p$ values per turbine.
    fmt : str, default="png"
        Output image format extension (`"png"`, `"pdf"`, `"svg"`).
    dpi : int, default=300
        Rasterization resolution in dots per inch.
    """
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for t in range(num_turbines):
        cp_t = cp_vec[:, t]
        mask = cp_t >= -1.0
        idx_sort = np.argsort(tsr_vec[mask])
        ax.plot(
            tsr_vec[mask][idx_sort],
            cp_t[mask][idx_sort],
            marker="o",
            label=f"Turbine {t+1}"
        )

    if num_turbines > 1:
        avg_cp = np.mean(cp_vec, axis=1)
        mask_avg = avg_cp >= -1.0
        idx_sort_avg = np.argsort(tsr_vec[mask_avg])
        ax.plot(
            tsr_vec[mask_avg][idx_sort_avg],
            avg_cp[mask_avg][idx_sort_avg],
            "--",
            color="black",
            linewidth=2,
            label="System Average"
        )

    ax.set_xlabel(r"TSR ($\lambda$)")
    ax.set_ylabel(r"$C_p$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()

    out_path = case_dir / f"cp_curve.{fmt.lower()}"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def export_coupled_case_results(
    case_name: str,
    config: dict[str, Any],
    turbines: list[Turbine],
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray,
    ct_vec: np.ndarray,
    rp_vec: np.ndarray,
    tp_vec: np.ndarray,
    zp_vec: np.ndarray,
    base_results_dir: str | Path = DEFAULT_RESULTS_DIR
) -> Path | None:
    """
    Orchestrate full export pipeline for a completed simulation case.

    Creates output directory structures, dumps the current YAML configuration snapshot,
    writes raw numerical data files for each turbine, and generates graphic plots
    based on the settings specified in the configuration dictionary.

    Parameters
    ----------
    case_name : str
        Unique descriptive folder name for the simulation run.
    config : dict of {str : Any}
        Active simulation configuration dictionary.
    turbines : list of Turbine
        List of initialized `Turbine` instances.
    tsr_vec : np.ndarray, shape (N_tsr,)
        Evaluated Tip Speed Ratio points array.
    cp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Power Coefficient matrix.
    ct_vec : np.ndarray, shape (N_tsr, N_turbines)
        Thrust Coefficient matrix.
    rp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Radial Force Coefficient matrix.
    tp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Tangential Force Coefficient matrix.
    zp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Axial Force Coefficient matrix.
    base_results_dir : str or Path, default="results"
        Base directory path where case directories are created.

    Returns
    -------
    Path or None
        Path object pointing to the output case directory if saving was enabled,
        or `None` if `output.save` was set to `False`.
    """
    output_cfg = config.get("output", {})

    # Check if export is globally enabled
    if not output_cfg.get("save", True):
        return None

    save_config_flag = output_cfg.get("save_config", True)
    save_plot_flag = output_cfg.get("save_plot", True)

    data_file_cfg = output_cfg.get("data_file", {})
    data_fmt = data_file_cfg.get("format", "dat")
    inc_header = data_file_cfg.get("include_header", True)

    plot_img_cfg = output_cfg.get("plot_image", {})
    img_fmt = plot_img_cfg.get("format", "png")
    img_dpi = int(plot_img_cfg.get("dpi", 300))

    case_dir = Path(base_results_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dump configuration YAML snapshot
    if save_config_flag:
        with open(case_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    num_turbines = len(turbines)

    # 2. Export raw tabular dataset
    _save_raw_data(
        case_dir=case_dir,
        num_turbines=num_turbines,
        tsr_vec=tsr_vec,
        cp_vec=cp_vec,
        ct_vec=ct_vec,
        rp_vec=rp_vec,
        tp_vec=tp_vec,
        zp_vec=zp_vec,
        fmt=data_fmt,
        include_header=inc_header
    )

    # 3. Generate layout and performance figures
    if save_plot_flag:
        plot_turbine_layout(turbines, case_dir, fmt=img_fmt, dpi=img_dpi)
        _plot_performance_curves(case_dir, num_turbines, tsr_vec, cp_vec, fmt=img_fmt, dpi=img_dpi)

    return case_dir

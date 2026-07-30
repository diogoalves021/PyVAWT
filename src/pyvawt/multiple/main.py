"""
Aerodynamic Simulation Runner for Coupled VAWT Turbines.

Orchestrates multi-turbine aerodynamic simulations using Actuator Cylinder theory
and NeuralFoil/Classic airfoil lookup adapters.
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from numba import njit

from src.pyvawt.multiple.data_generation import get_cl_cd_neuralfoil, load_config
from src.pyvawt.multiple.read_data import readaerodyn
from src.pyvawt.multiple.simulation import Environment, Turbine, actuatorcylinder
from src.pyvawt.ui.ui import UI, MultiTurbineUI

# Logging configuration
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Simulation Default Constants
DEFAULT_TSR_START: float = 1.0
DEFAULT_TSR_END: float = 7.0
DEFAULT_TSR_POINTS: int = 20
DEFAULT_RESULTS_DIR: str = "results"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class SimulationContext(NamedTuple):
    """
    Structured container for initialized simulation physical components.

    Attributes
    ----------
    turbines : list[Turbine]
        List of instantiated `Turbine` physical objects.
    env : Environment
        Instantiated simulation `Environment` (fluid properties).
    simulation_params : dict[str, Any]
        Raw solver configuration settings.
    turbine_params : dict[str, Any]
        Raw turbine physical geometry parameters.
    environment_params : dict[str, Any]
        Raw environment fluid dynamic parameters.
    radius : float
        Rotor radius in meters ($m$).
    ntheta : int
        Number of azimuthal discretizations along the rotor boundary.
    """
    turbines: list[Turbine]
    env: Environment
    simulation_params: dict[str, Any]
    turbine_params: dict[str, Any]
    environment_params: dict[str, Any]
    radius: float
    ntheta: int


# ==============================================================================
# HIGH-PERFORMANCE INTERPOLATION KERNELS
# ==============================================================================

@njit(fastmath=True, cache=True)
def _bilinear_interp_2d_numba(
    alpha_wrapped: np.ndarray,
    w_clamped: np.ndarray,
    values: np.ndarray,
    alpha_min: float,
    inv_dalpha: float,
    log_w_min: float,
    inv_dlog_w: float,
    n_alpha: int,
    n_w: int
) -> np.ndarray:
    """
    High-performance C-level compiled 2D bilinear interpolation kernel via Numba.

    Performs zero-allocation bound-clamped lookup on regular logarithmic grids.

    Parameters
    ----------
    alpha_wrapped : np.ndarray
        1D or 2D array of wrapped angles of attack in radians ($[-\pi, \pi]$).
    w_clamped : np.ndarray
        Array of non-dimensional flow parameters or Reynolds proxies.
    values : np.ndarray
        2D lookup array of shape `(n_alpha, n_w)` holding aerodynamic
        coefficients ($C_l$ or $C_d$).
    alpha_min : float
        Minimum angle of attack bound on the regular grid in radians.
    inv_dalpha : float
        Inverse step size ($1 / \Delta\alpha$) along the angle-of-attack grid.
    log_w_min : float
        Natural logarithm of the minimum value bound for `w`.
    inv_dlog_w : float
        Inverse step size along the logarithmic `w` grid.
    n_alpha : int
        Number of grid points along the angle-of-attack dimension.
    n_w : int
        Number of grid points along the logarithmic `w` dimension.

    Returns
    -------
    np.ndarray
        Interpolated values with identical shape to `alpha_wrapped`.
    """
    alpha_flat = alpha_wrapped.ravel()
    w_flat = w_clamped.ravel()
    n = alpha_flat.size
    out = np.empty(n, dtype=np.float64)

    max_i = n_alpha - 1.000001
    max_j = n_w - 1.000001

    for k in range(n):
        fi = (alpha_flat[k] - alpha_min) * inv_dalpha
        fj = (np.log(w_flat[k]) - log_w_min) * inv_dlog_w

        # Bounds clamping
        fi = max(0.0, min(fi, max_i))
        fj = max(0.0, min(fj, max_j))

        i0 = int(fi)
        j0 = int(fj)
        i1 = i0 + 1
        j1 = j0 + 1

        t = fi - i0
        u = fj - j0

        v00 = values[i0, j0]
        v10 = values[i1, j0]
        v01 = values[i0, j1]
        v11 = values[i1, j1]

        out[k] = (1.0 - t) * (1.0 - u) * v00 + t * (1.0 - u) * v10 + (1.0 - t) * u * v01 + t * u * v11

    return out.reshape(alpha_wrapped.shape)


class FastBilinear2D:
    """
    Python wrapper interface for the Numba-compiled 2D bilinear interpolator.

    Pre-computes lookup parameters to enable $O(1)$ interpolation throughput.

    Attributes
    ----------
    n_alpha : int
        Number of angular discretization grid points.
    n_w : int
        Number of flow parameter discretization grid points.
    values : np.ndarray
        Contiguous C-memory array holding grid response values.
    alpha_min : float
        Minimum grid angle limit.
    inv_dalpha : float
        Scale factor for indexing along the alpha grid.
    log_w_min : float
        Logarithmic minimum bound for $w$.
    inv_dlog_w : float
        Scale factor for logarithmic indexing along $w$.
    """

    def __init__(self, alpha_grid: np.ndarray, w_grid: np.ndarray, values: np.ndarray) -> None:
        """
        Initialize grid boundaries and indexing scale factors.

        Parameters
        ----------
        alpha_grid : np.ndarray
            Linearly spaced 1D grid array of angles of attack ($\alpha$).
        w_grid : np.ndarray
            Geometrically spaced 1D grid array of flow state parameters ($w$).
        values : np.ndarray
            2D array of grid properties matching shape `(len(alpha_grid), len(w_grid))`.
        """
        self.n_alpha = len(alpha_grid)
        self.n_w = len(w_grid)
        self.values = np.ascontiguousarray(values, dtype=np.float64)

        self.alpha_min = float(alpha_grid[0])
        self.inv_dalpha = float((self.n_alpha - 1) / (alpha_grid[-1] - alpha_grid[0]))

        self.log_w_min = float(np.log(w_grid[0]))
        self.inv_dlog_w = float((self.n_w - 1) / (np.log(w_grid[-1]) - np.log(w_grid[0])))

    def __call__(self, alpha_wrapped: np.ndarray, w_clamped: np.ndarray) -> np.ndarray:
        """
        Evaluate interpolated grid values for given alpha and w query arrays.

        Parameters
        ----------
        alpha_wrapped : np.ndarray
            Wrapped angle array in radians.
        w_clamped : np.ndarray
            Query state parameter array.

        Returns
        -------
        np.ndarray
            Interpolated scalar or array matching query shape.
        """
        return _bilinear_interp_2d_numba(
            alpha_wrapped,
            w_clamped,
            self.values,
            self.alpha_min,
            self.inv_dalpha,
            self.log_w_min,
            self.inv_dlog_w,
            self.n_alpha,
            self.n_w
        )


class NeuralFoilAirfoilWrapper:
    """
    Optimized airfoil polar evaluation adapter backed by disk-cached Look-Up Tables.

    Attributes
    ----------
    turbine_index : int
        Zero-based index identifying the target turbine in array.
    airfoil_index : int
        Zero-based index identifying the aerodynamic profile.
    chord : float
        Blade chord length in meters ($m$).
    rho : float
        Fluid density in $kg/m^3$.
    mu : float
        Dynamic fluid viscosity in $Pa \cdot s$.
    alpha_grid : np.ndarray
        Pre-generated 1D angular grid.
    w_grid : np.ndarray
        Pre-generated 1D flow parameter grid.
    cl_grid : np.ndarray
        Pre-computed 2D lift coefficient grid.
    cd_grid : np.ndarray
        Pre-computed 2D drag coefficient grid.
    """

    def __init__(
        self, 
        turbine_index: int, 
        airfoil_index: int, 
        n_alpha: int = 1800, 
        n_w: int = 40
    ) -> None:
        """
        Initialize wrapper, generating or loading cached polars from disk.

        Parameters
        ----------
        turbine_index : int
            Index of the turbine within array system.
        airfoil_index : int
            Index of the selected aerodynamic profile.
        n_alpha : int, optional
            Resolution of angular grid, by default 1800.
        n_w : int, optional
            Resolution of flow grid, by default 40.
        """
        self.turbine_index = turbine_index
        self.airfoil_index = airfoil_index

        config = load_config()
        chord_entry = config["turbine"]["chord"]

        if isinstance(chord_entry, list):
            self.chord = float(chord_entry[self.turbine_index % len(chord_entry)])
        else:
            self.chord = float(chord_entry)

        self.rho = float(config["environment"]["rho"])
        self.mu = float(config["environment"]["mu"])

        cache_dir = Path(".cache_lut")
        cache_dir.mkdir(exist_ok=True)

        cache_sig = f"t{turbine_index}_a{airfoil_index}_c{self.chord:.6f}_r{self.rho:.4f}_m{self.mu:.4e}_na{n_alpha}_nw{n_w}"
        hash_id = hashlib.md5(cache_sig.encode("utf-8")).hexdigest()[:10]
        cache_file = cache_dir / f"lut_{hash_id}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file)
                self.alpha_grid = data["alpha_grid"]
                self.w_grid = data["w_grid"]
                self.cl_grid = data["cl_grid"]
                self.cd_grid = data["cd_grid"]
            except Exception:
                self._build_and_cache_lut(n_alpha, n_w, cache_file)
        else:
            self._build_and_cache_lut(n_alpha, n_w, cache_file)

        self._interp_cl = FastBilinear2D(self.alpha_grid, self.w_grid, self.cl_grid)
        self._interp_cd = FastBilinear2D(self.alpha_grid, self.w_grid, self.cd_grid)

    def _build_and_cache_lut(self, n_alpha: int, n_w: int, cache_file: Path) -> None:
        """
        Evaluate NeuralFoil model over grid coordinates and compress to disk.

        Parameters
        ----------
        n_alpha : int
            Angular grid point count.
        n_w : int
            Flow parameter grid point count.
        cache_file : Path
            Target compressed file path (`.npz`).
        """
        self.alpha_grid = np.linspace(-np.pi, np.pi, n_alpha)
        self.w_grid = np.geomspace(0.1, 150.0, n_w)

        alpha_mesh, w_mesh = np.meshgrid(self.alpha_grid, self.w_grid, indexing="ij")
        cl_flat, cd_flat = get_cl_cd_neuralfoil(
            alpha_mesh.ravel(),
            w_mesh.ravel(),
            self.turbine_index,
            self.airfoil_index
        )

        self.cl_grid = np.asarray(cl_flat, dtype=np.float64).reshape(alpha_mesh.shape)
        self.cd_grid = np.asarray(cd_flat, dtype=np.float64).reshape(alpha_mesh.shape)

        np.savez_compressed(
            cache_file,
            alpha_grid=self.alpha_grid,
            w_grid=self.w_grid,
            cl_grid=self.cl_grid,
            cd_grid=self.cd_grid
        )

    def get_coefficients(
        self, 
        alpha_rad: float | np.ndarray, 
        Re: float | np.ndarray | None = None
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Evaluate $C_l$ and $C_d$ coefficients via $O(1)$ grid interpolation.

        Parameters
        ----------
        alpha_rad : float or np.ndarray
            Angle of attack in radians.
        Re : float or np.ndarray, optional
            Reynolds number query array or scalar, by default None.

        Returns
        -------
        tuple[float | np.ndarray, float | np.ndarray]
            Lift coefficient ($C_l$) and Drag coefficient ($C_d$).
        """
        alpha_arr = np.asarray(alpha_rad, dtype=np.float64)
        alpha_wrapped = (alpha_arr + np.pi) % (2.0 * np.pi) - np.pi

        if Re is None:
            w_arr = np.full_like(alpha_wrapped, 10.0)
        else:
            w_arr = np.asarray(Re, dtype=np.float64) * (self.mu / (self.rho * self.chord))

        w_clamped = np.clip(w_arr, self.w_grid[0], self.w_grid[-1])

        cl = self._interp_cl(alpha_wrapped, w_clamped)
        cd = self._interp_cd(alpha_wrapped, w_clamped)

        if alpha_arr.ndim == 0:
            return float(cl), float(cd)

        return cl, cd

    def __call__(self, alpha_rad: float | np.ndarray, Re: float | np.ndarray | None = None) -> tuple[Any, Any]:
        """
        Callable alias for `get_coefficients`.

        Parameters
        ----------
        alpha_rad : float or np.ndarray
            Angle of attack in radians.
        Re : float or np.ndarray, optional
            Reynolds number query array or scalar, by default None.

        Returns
        -------
        tuple[Any, Any]
            Lift and drag coefficients.
        """
        return self.get_coefficients(alpha_rad, Re)


# ==============================================================================
# EXPORT AND PLOTTING UTILITIES
# ==============================================================================

def _apply_plot_style() -> None:
    """Apply standardized typography styles for technical plots."""
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
    Generate a top-view spatial layout plot of the turbine array.

    Parameters
    ----------
    turbines : list[Turbine]
        List of instantiated `Turbine` objects.
    case_dir : Path
        Target directory path for image export.
    fmt : str, optional
        File format extension (e.g., 'png', 'pdf', 'svg'), by default "png".
    dpi : int, optional
        Image resolution in dots per inch, by default 300.
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
    Export raw numerical simulation results to structured tabular files.

    Parameters
    ----------
    case_dir : Path
        Target output directory path.
    num_turbines : int
        Number of turbines in simulation system.
    tsr_vec : np.ndarray
        1D array of evaluated Tip Speed Ratios ($\lambda$).
    cp_vec : np.ndarray
        Matrix of Power Coefficients of shape `(n_points, n_turbines)`.
    ct_vec : np.ndarray
        Matrix of Thrust Coefficients of shape `(n_points, n_turbines)`.
    rp_vec : np.ndarray
        Radial force results matrix.
    tp_vec : np.ndarray
        Tangential force results matrix.
    zp_vec : np.ndarray
        Axial/Vertical force results matrix.
    fmt : str, optional
        File format extension ('dat' or 'csv'), by default "dat".
    include_header : bool, optional
        If True, writes standard column names as header, by default True.
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
    Generate aerodynamic performance curves ($C_p$ vs. TSR).

    Parameters
    ----------
    case_dir : Path
        Output directory path.
    num_turbines : int
        Number of active turbines.
    tsr_vec : np.ndarray
        Evaluated TSR point vector.
    cp_vec : np.ndarray
        Computed Power Coefficients array.
    fmt : str, optional
        Output image extension, by default "png".
    dpi : int, optional
        Target image rendering DPI, by default 300.
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
    Save all outputs and configuration files as specified in the YAML config.

    Parameters
    ----------
    case_name : str
        Identifier name for the target simulation run directory.
    config : dict[str, Any]
        Simulation configuration dictionary.
    turbines : list[Turbine]
        List of evaluated `Turbine` instances.
    tsr_vec : np.ndarray
        TSR sweep array.
    cp_vec : np.ndarray
        Power coefficient matrix.
    ct_vec : np.ndarray
        Thrust coefficient matrix.
    rp_vec : np.ndarray
        Radial force matrix.
    tp_vec : np.ndarray
        Tangential force matrix.
    zp_vec : np.ndarray
        Axial force matrix.
    base_results_dir : str or Path, optional
        Root output directory path, by default `DEFAULT_RESULTS_DIR`.

    Returns
    -------
    Path or None
        Resolved path to case results directory if saved, or None if save disabled.
    """
    output_cfg = config.get("output", {})

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

    if save_config_flag:
        with open(case_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    num_turbines = len(turbines)
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

    if save_plot_flag:
        plot_turbine_layout(turbines, case_dir, fmt=img_fmt, dpi=img_dpi)
        _plot_performance_curves(case_dir, num_turbines, tsr_vec, cp_vec, fmt=img_fmt, dpi=img_dpi)

    return case_dir


# ==============================================================================
# CORE SOLVER & SIMULATION ORCHESTRATION
# ==============================================================================

def _execute_tsr_sweep(
    turbines: list[Turbine],
    env: Environment,
    ntheta: int,
    tsr_vec: np.ndarray,
    var_omega_vinf: int,
    vinf: float,
    radius: float,
    num_turbines: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Execute sequential TSR sweep iterations while triggering progress bar updates.

    Parameters
    ----------
    turbines : list[Turbine]
        List of target `Turbine` physical objects.
    env : Environment
        Instantiated fluid `Environment`.
    ntheta : int
        Azimuthal discretization step count.
    tsr_vec : np.ndarray
        Array of TSR sweep values to simulate.
    var_omega_vinf : int
        Flag for variable strategy (0: vary Omega, 1: vary Vinf).
    vinf : float
        Free stream velocity in $m/s$.
    radius : float
        Rotor radius in meters ($m$).
    num_turbines : int
        Number of coupled turbines.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]
        Tuple containing (cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec, warnings).

    Raises
    ------
    ValueError
        If `var_omega_vinf` is not 0 or 1.
    """
    n_points = len(tsr_vec)
    cp_vec = np.zeros((n_points, num_turbines))
    ct_vec = np.zeros((n_points, num_turbines))
    rp_vec = np.zeros((n_points, num_turbines))
    tp_vec = np.zeros((n_points, num_turbines))
    zp_vec = np.zeros((n_points, num_turbines))
    theta_vec = np.zeros((n_points, ntheta))

    w_guess = None
    warnings: list[str] = []

    UI.section("SWEEP EXECUTION")
    sweep_start_time = time.perf_counter()

    for i, tsr in enumerate(tsr_vec):
        if var_omega_vinf == 0:
            for turbine in turbines:
                turbine.Omega = vinf * tsr / radius
        elif var_omega_vinf == 1:
            for turbine in turbines:
                turbine.Omega = 13.62 * 2 * np.pi / 60.0
            env.Vinf = turbines[0].Omega * radius / tsr
        else:
            raise ValueError(f"Invalid omega/vinf strategy configuration: {var_omega_vinf}")

        res = actuatorcylinder(turbines, env, ntheta, w0=w_guess)

        if len(res) == 8:
            ct, cp, rp, tp, zp, theta, w_guess, solver_warns = res
            if solver_warns:
                warnings.extend(solver_warns)
        else:
            ct, cp, rp, tp, zp, theta, w_guess = res

        for t in range(num_turbines):
            cp_vec[i, t] = cp[t]
            ct_vec[i, t] = ct[t]

            if rp.ndim == 2:
                rp_vec[i, t] = rp[0, t]
                tp_vec[i, t] = tp[0, t]
                zp_vec[i, t] = zp[0, t]
            else:
                rp_vec[i, t] = rp[t]
                tp_vec[i, t] = tp[t]
                zp_vec[i, t] = zp[t]

        theta_vec[i, :] = theta

        elapsed = time.perf_counter() - sweep_start_time
        MultiTurbineUI.print_progress(i + 1, n_points, elapsed)

    return cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec, warnings


def initialize_turbine_and_environment(config: dict[str, Any]) -> SimulationContext:
    """
    Build physical Turbine and Environment instances from configuration dict.

    Parameters
    ----------
    config : dict[str, Any]
        Loaded simulation YAML dictionary.

    Returns
    -------
    SimulationContext
        Container holding initialized objects and physical parameters.

    Raises
    ------
    KeyError
        If mandatory configuration fields (e.g., file paths) are missing.
    """
    turbine_params = config["turbine"]
    environment_params = config["environment"]
    solver_params = config.get("solver", {})

    r = float(turbine_params["r"])
    twist = float(turbine_params["twist"])
    delta = float(turbine_params["delta"])
    b_blades = int(turbine_params["B"])
    center_x = turbine_params["centerX"]
    center_y = turbine_params["centerY"]
    omega = float(turbine_params["Omega"])
    ntheta = int(turbine_params["ntheta"])

    vinf_raw = environment_params["Vinf"]
    vinf = float(vinf_raw[0]) if isinstance(vinf_raw, list) else float(vinf_raw)
    rho = float(environment_params["rho"])
    mu = float(environment_params["mu"])

    num_turbines = int(solver_params.get("num_turbines", 1))

    raw_chord = turbine_params["chord"]
    chord_list = [float(c) for c in raw_chord] if isinstance(raw_chord, list) else [float(raw_chord)] * num_turbines
    if len(chord_list) < num_turbines:
        chord_list.extend([chord_list[-1]] * (num_turbines - len(chord_list)))

    method = solver_params.get("method", "neuralfoil")
    use_neuralfoil = (method == "neuralfoil")
    turbines_airfoils = []

    if use_neuralfoil:
        for i in range(num_turbines):
            wrapper = NeuralFoilAirfoilWrapper(turbine_index=i, airfoil_index=i)
            turbines_airfoils.append(wrapper)
    else:
        file_cfg = solver_params.get("file", {})
        aero_profile = file_cfg.get("path")
        if not aero_profile:
            raise KeyError("Airfoil file path missing under 'solver.file.path'")
        classic_profile = readaerodyn(aero_profile)
        turbines_airfoils = [classic_profile] * num_turbines

    cx_list = [float(center_x) + i * 4.0 * r for i in range(num_turbines)] if isinstance(center_x, (int, float)) else [float(x) for x in center_x]
    cy_list = [float(center_y) for _ in range(num_turbines)] if isinstance(center_y, (int, float)) else [float(y) for y in center_y]

    if len(cx_list) < num_turbines:
        cx_list.extend([cx_list[-1] + 4.0 * r for _ in range(num_turbines - len(cx_list))])
    if len(cy_list) < num_turbines:
        cy_list.extend([cy_list[-1] for _ in range(num_turbines - len(cy_list))])

    turbines = [
        Turbine(r, chord_list[i], twist, delta, b_blades, turbines_airfoils[i], omega, cx_list[i], cy_list[i])
        for i in range(num_turbines)
    ]

    env = Environment(vinf, rho, mu)

    return SimulationContext(
        turbines=turbines,
        env=env,
        simulation_params=solver_params,
        turbine_params=turbine_params,
        environment_params=environment_params,
        radius=r,
        ntheta=ntheta
    )


def run_simulation_case(params: tuple[int, int, float, float, float]) -> dict[str, Any]:
    """
    Run a single simulation case and export performance reports and plots.

    Parameters
    ----------
    params : tuple[int, int, float, float, float]
        Tuple containing case metadata `(case_idx, total_cases, chord, solidity, vinf)`.

    Returns
    -------
    dict[str, Any]
        Summary status dictionary containing execution metadata.

    Raises
    ------
    Exception
        Re-raises any critical exception encountered during execution after logging.
    """
    _, _, chord, solidity, vinf = params
    config = load_config()

    solver_cfg = config.get("solver", {})
    method = solver_cfg.get("method", "neuralfoil")
    use_neuralfoil = (method == "neuralfoil")

    if use_neuralfoil:
        nf_cfg = solver_cfg.get("neuralfoil", {})
        raw_airfoil = nf_cfg.get("airfoil", ["naca0018"])
        airfoil_name = raw_airfoil[0] if isinstance(raw_airfoil, list) else str(raw_airfoil)
    else:
        file_cfg = solver_cfg.get("file", {})
        airfoil_file = file_cfg.get("path")
        if not airfoil_file:
            raise KeyError("Classic profile path missing in 'solver.file.path'")
        airfoil_name = Path(airfoil_file).stem

    num_turbines = int(solver_cfg.get("num_turbines", 1))
    blades = int(config.get("turbine", {}).get("B", 3))
    radius = round(chord * blades / solidity, 4)

    fixed_param = str(solver_cfg.get("fixed_parameter", "vinf")).lower()
    var_omega_vinf = 0 if fixed_param == "vinf" else 1

    config["turbine"]["chord"] = [chord] * num_turbines
    config["turbine"]["solidity"] = [solidity] * num_turbines
    config["environment"]["Vinf"] = [vinf] if isinstance(config["environment"]["Vinf"], list) else vinf
    config["turbine"]["r"] = radius

    case_name = f"{airfoil_name}_turb{num_turbines}_b{blades}_r{radius}_ch{chord}_sol{solidity}_vinf{vinf}".replace(".", "p")

    t_start_setup = time.perf_counter()
    context = initialize_turbine_and_environment(config)
    jit_setup_time = time.perf_counter() - t_start_setup

    MultiTurbineUI.print_init(turbines=context.turbines, jit_time=jit_setup_time, mode_coupled=True)

    start_time = time.perf_counter()

    try:
        tsr_vec = np.linspace(DEFAULT_TSR_START, DEFAULT_TSR_END, DEFAULT_TSR_POINTS)

        cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec, warnings = _execute_tsr_sweep(
            turbines=context.turbines,
            env=context.env,
            ntheta=context.ntheta,
            tsr_vec=tsr_vec,
            var_omega_vinf=var_omega_vinf,
            vinf=vinf,
            radius=context.radius,
            num_turbines=num_turbines
        )

        elapsed = time.perf_counter() - start_time

        case_dir = export_coupled_case_results(
            case_name=case_name,
            config=config,
            turbines=context.turbines,
            tsr_vec=tsr_vec,
            cp_vec=cp_vec,
            ct_vec=ct_vec,
            rp_vec=rp_vec,
            tp_vec=tp_vec,
            zp_vec=zp_vec,
            base_results_dir=DEFAULT_RESULTS_DIR
        )

        output_dir_str = str(case_dir.resolve()) if case_dir else "Disabled (save=false)"

        MultiTurbineUI.print_results(
            turbines=context.turbines,
            total_time=elapsed,
            output_dir=output_dir_str,
            cp_results=cp_vec.T,
            ct_results=ct_vec.T,
            tsr_vec=tsr_vec,
            warnings=warnings
        )

        return {
            "name": case_name,
            "airfoil": airfoil_name,
            "num_turbines": num_turbines,
            "blades": blades,
            "radius": radius,
            "chord": chord,
            "solidity": solidity,
            "vinf": vinf,
            "status": "OK",
            "time_sec": round(elapsed, 2)
        }

    except Exception as err:
        logger.error(f"Execution failed for case '{case_name}': {err}", exc_info=True)
        raise


def main() -> None:
    """Main execution entrypoint for running coupled turbine simulations."""
    MultiTurbineUI.print_header()

    try:
        config = load_config()

        raw_chord = config["turbine"]["chord"]
        raw_solidity = config["turbine"]["solidity"]
        raw_vinf = config["environment"]["Vinf"]

        chord_val = float(raw_chord[0] if isinstance(raw_chord, list) else raw_chord)
        solidity_val = float(raw_solidity[0] if isinstance(raw_solidity, list) else raw_solidity)
        vinf_val = float(raw_vinf[0] if isinstance(raw_vinf, list) else raw_vinf)

        run_simulation_case((0, 0, chord_val, solidity_val, vinf_val))

    except Exception as err:
        logger.critical(f"Fatal error during pipeline execution: {err}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

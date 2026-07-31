"""
Aerodynamic Properties and Airfoil Evaluation Module for PyVAWT.

Provides dataset parsing interfaces for classic AeroDyn files, NeuralFoil ML model
wrappers via AeroSandbox, and high-performance 2D lookup table (LUT) disk-caching adapters.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import aerosandbox as asb
import numpy as np
from scipy.interpolate import UnivariateSpline

from src.pyvawt.multiple.interpolation import FastBilinear2D
from src.pyvawt.multiple.settings import load_config

logger = logging.getLogger(__name__)

# Generic type alias accepting both scalar floats and NumPy multidimensional arrays
ArrayOrFloat = float | np.ndarray


def readaerodyn(
    filename: str | Path,
    s_cl: float = 0.1,
    s_cd: float = 0.001,
    skip_header: int = 13,
) -> Callable[[ArrayOrFloat], tuple[ArrayOrFloat, ArrayOrFloat]]:
    """
    Parse an AeroDyn polar file and build 1D smoothing spline interpolators.

    Reads tabular angle of attack ($\alpha$), lift ($C_l$), and drag ($C_d$) data from an
    AeroDyn formatted file, constructing 1D `UnivariateSpline` interpolators.

    Parameters
    ----------
    filename : str or Path
        Path to the target AeroDyn airfoil polar text file.
    s_cl : float, default=0.1
        Spline smoothing factor for lift coefficient ($C_l$).
    s_cd : float, default=0.001
        Spline smoothing factor for drag coefficient ($C_d$).
    skip_header : int, default=13
        Number of header lines to skip before numerical polar data begins.

    Returns
    -------
    af : Callable[[ArrayOrFloat], tuple[ArrayOrFloat, ArrayOrFloat]]
        Evaluation callback function accepting angle of attack $\alpha$ in radians and returning
        a tuple containing interpolated $(C_l, C_d)$ values.

    Raises
    ------
    FileNotFoundError
        If the specified polar file path does not exist on disk.
    """
    filepath = Path(filename)

    if not filepath.is_file():
        raise FileNotFoundError(f"AeroDyn polar file not found: {filepath.resolve()}")

    alpha_deg: list[float] = []
    cl_list: list[float] = []
    cd_list: list[float] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        # Skip header lines
        for _ in range(skip_header):
            next(f, None)

        # Read data rows until encountering the 'EOT' termination line
        for line in f:
            if "EOT" in line:
                break

            parts = line.split()
            if len(parts) >= 3:
                alpha_deg.append(float(parts[0]))
                cl_list.append(float(parts[1]))
                cd_list.append(float(parts[2]))

    # Convert angle of attack from degrees to radians
    alpha_rad = np.deg2rad(alpha_deg)
    cl_arr = np.array(cl_list, dtype=np.float64)
    cd_arr = np.array(cd_list, dtype=np.float64)

    # Ensure strictly increasing coordinate order (required by UnivariateSpline)
    sort_idx = np.argsort(alpha_rad)
    alpha_sorted = alpha_rad[sort_idx]
    cl_sorted = cl_arr[sort_idx]
    cd_sorted = cd_arr[sort_idx]

    # Instantiate 1D splines
    afcl = UnivariateSpline(alpha_sorted, cl_sorted, s=s_cl)
    afcd = UnivariateSpline(alpha_sorted, cd_sorted, s=s_cd)

    def af(alpha: ArrayOrFloat) -> tuple[ArrayOrFloat, ArrayOrFloat]:
        """
        Evaluate spline interpolated lift and drag coefficients.

        Parameters
        ----------
        alpha : float or np.ndarray
            Angle of attack in radians.

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray)
            Evaluated $(C_l, C_d)$ coefficients matching input shape.
        """
        return afcl(alpha), afcd(alpha)

    return af


@lru_cache(maxsize=16)
def _get_airfoil_instance(name: str) -> asb.Airfoil:
    """
    Cache AeroSandbox Airfoil objects to eliminate redundant instantiation overhead.

    Parameters
    ----------
    name : str
        Airfoil profile name identifier (e.g., `"naca0018"`).

    Returns
    -------
    asb.Airfoil
        Cached AeroSandbox `Airfoil` instance.
    """
    return asb.Airfoil(name=name)


def get_cl_cd_neuralfoil(
    alpha: ArrayOrFloat,
    W: ArrayOrFloat,
    turbine_index: int = 0,
    airfoil_index: int = 0,
    config_path: str | Path = "src/pyvawt/config/config_multiple.yaml",
) -> tuple[ArrayOrFloat, ArrayOrFloat]:
    """
    Evaluate lift ($C_l$) and drag ($C_d$) coefficients using NeuralFoil via AeroSandbox.

    Calculates local flow conditions (Reynolds number $Re$, Mach number $M$) based on
    fluid properties ($\rho$, $\mu$) and queries the NeuralFoil neural network backend.

    Parameters
    ----------
    alpha : float or np.ndarray
        Angle of attack in radians.
    W : float or np.ndarray
        Local apparent wind speed magnitude ($W$) in m/s.
    turbine_index : int, default=0
        Index of target turbine to pull chord length $c$ from configuration.
    airfoil_index : int, default=0
        Index of target airfoil profile to select geometry profile name from configuration.
    config_path : str or Path, default="src/pyvawt/config/config_multiple.yaml"
        Path to simulation YAML configuration file.

    Returns
    -------
    cl : float or np.ndarray
        Evaluated lift coefficient(s) $C_l$.
    cd : float or np.ndarray
        Evaluated drag coefficient(s) $C_d$.
    """
    config = load_config(config_path)

    # Extract geometric and physical properties
    chords = config["turbine"]["chord"]
    chord = float(chords[turbine_index % len(chords)]) if isinstance(chords, list) else float(chords)

    rho = float(config["environment"]["rho"])
    mu = float(config["environment"]["mu"])

    solver_cfg = config.get("solver", {})
    nf_cfg = solver_cfg.get("neuralfoil", {})

    airfoils = nf_cfg.get("airfoil", ["naca0018"])
    airfoil_name = str(airfoils[airfoil_index % len(airfoils)]) if isinstance(airfoils, list) else str(airfoils)

    model_size = nf_cfg.get("model_size", "large")
    include_360 = nf_cfg.get("include_360_deg_effects", True)

    # Calculate dimensionless physical fluid dynamics parameters
    Re = rho * W * chord / mu
    mach = W / 343.2  # Speed of sound in ambient air (~343.2 m/s)

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

    # Preserve scalar return type if inputs were pure scalars
    if np.isscalar(alpha) and np.isscalar(W):
        return float(cl_res), float(cd_res)

    return cl_res, cd_res


class NeuralFoilAirfoilWrapper:
    """
    High-performance airfoil evaluation wrapper backed by disk-cached LUTs.

    Pre-computes a 2D structured mesh over angle of attack ($\alpha \in [-\pi, \pi]$)
    and logarithmic relative velocity ($w \in [0.1, 150.0]$ m/s). Caches generated matrices
    to `.npz` files using MD5 parameter hashing and evaluates runtime queries using
    fast Numba 2D bilinear interpolation (`FastBilinear2D`).

    Parameters
    ----------
    turbine_index : int
        Index identifier of the turbine instance.
    airfoil_index : int
        Index identifier of the airfoil geometry profile.
    n_alpha : int, default=1800
        Number of discretization points along the $\alpha$ grid axis.
    n_w : int, default=40
        Number of discretization points along the logarithmic relative velocity ($w$) grid axis.

    Attributes
    ----------
    chord : float
        Rotor blade chord length in meters.
    rho : float
        Fluid density ($\rho$) in $\text{kg/m}^3$.
    mu : float
        Dynamic viscosity ($\mu$) in $\text{Pa}\cdot\text{s}$.
    alpha_grid : np.ndarray, shape (n_alpha,)
        Linear coordinate grid for $\alpha$.
    w_grid : np.ndarray, shape (n_w,)
        Logarithmic coordinate grid for relative flow velocity $w$.
    cl_grid : np.ndarray, shape (n_alpha, n_w)
        Pre-computed 2D lookup matrix for lift coefficients ($C_l$).
    cd_grid : np.ndarray, shape (n_alpha, n_w)
        Pre-computed 2D lookup matrix for drag coefficients ($C_d$).
    """

    def __init__(
        self, 
        turbine_index: int, 
        airfoil_index: int, 
        n_alpha: int = 1800, 
        n_w: int = 40
    ) -> None:
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

        # Prepare disk cache path and MD5 digest identifier
        cache_dir = Path(".cache_lut")
        cache_dir.mkdir(exist_ok=True)

        cache_sig = f"t{turbine_index}_a{airfoil_index}_c{self.chord:.6f}_r{self.rho:.4f}_m{self.mu:.4e}_na{n_alpha}_nw{n_w}"
        hash_id = hashlib.md5(cache_sig.encode("utf-8")).hexdigest()[:10]
        cache_file = cache_dir / f"lut_{hash_id}.npz"

        # Load existing lookup cache or execute grid generator
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                self.alpha_grid = data["alpha_grid"]
                self.w_grid = data["w_grid"]
                self.cl_grid = data["cl_grid"]
                self.cd_grid = data["cd_grid"]
            except Exception as err:
                logger.warning(f"Failed to load cache file '{cache_file}'. Rebuilding LUT... Error: {err}")
                self._build_and_cache_lut(n_alpha, n_w, cache_file)
        else:
            self._build_and_cache_lut(n_alpha, n_w, cache_file)

        # Bind compiled 2D bilinear interpolation kernel instances
        self._interp_cl = FastBilinear2D(self.alpha_grid, self.w_grid, self.cl_grid)
        self._interp_cd = FastBilinear2D(self.alpha_grid, self.w_grid, self.cd_grid)

    def _build_and_cache_lut(self, n_alpha: int, n_w: int, cache_file: Path) -> None:
        """
        Construct 2D aerodynamic lookup tables and save compressed array to disk.

        Parameters
        ----------
        n_alpha : int
            Angular discretization resolution ($\alpha$).
        n_w : int
            Velocity discretization resolution ($w$).
        cache_file : Path
            Destination disk path for compressed `.npz` archive.
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
        Query interpolated lift ($C_l$) and drag ($C_d$) coefficients.

        Wraps the input angle of attack into $[-\pi, \pi]$, converts $Re$ to relative
        velocity magnitude $w$, clips values to lookup grid boundaries, and calls
        the 2D bilinear interpolator.

        Parameters
        ----------
        alpha_rad : float or np.ndarray
            Angle of attack in radians.
        Re : float or np.ndarray or None, default=None
            Reynolds number $Re$. If `None`, assumes a nominal velocity $w = 10.0$ m/s.

        Returns
        -------
        cl : float or np.ndarray
            Interpolated lift coefficient ($C_l$).
        cd : float or np.ndarray
            Interpolated drag coefficient ($C_d$).
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

    def __call__(
        self, 
        alpha_rad: float | np.ndarray, 
        Re: float | np.ndarray | None = None
    ) -> tuple[Any, Any]:
        """
        Callable interface shortcut forwarding to `get_coefficients`.

        Parameters
        ----------
        alpha_rad : float or np.ndarray
            Angle of attack in radians.
        Re : float or np.ndarray or None, default=None
            Reynolds number $Re$.

        Returns
        -------
        tuple of (float or np.ndarray, float or np.ndarray)
            Interpolated $(C_l, C_d)$ coefficients.
        """
        return self.get_coefficients(alpha_rad, Re)

"""
Aerodynamic profiles, file readers, surrogates, and high-level wrappers.

Provides unified interfaces for evaluating aerodynamic coefficients (Cl, Cd)
via file-based polars (QBlade/AeroDyn) or surrogate models (NeuralFoil) with
2D Look-Up Table (LUT) caching.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable
import numpy as np
import yaml
import aerosandbox as asb
from scipy.interpolate import UnivariateSpline

from src.pyvawt.single.interpolation import interpolate_2d_lut


# ==============================================================================
# GLOBAL RAM CACHE
# ==============================================================================

_AERO_LUT_CACHE: dict[
    tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
] = {}


# ==============================================================================
# POLAR READERS & GENERATORS
# ==============================================================================

def readaerodyn(
    filename: str
) -> Callable[[float | np.ndarray], tuple[np.ndarray | float, np.ndarray | float]]:
    """
    Read an aerodynamic polar file and return a spline interpolation function.

    Detects QBlade or AeroDyn (.dat) file formats, extracts angle of attack (deg),
    lift ($C_l$), and drag ($C_d$) coefficients, and builds 1D smoothed splines.

    Parameters
    ----------
    filename : str
        Path to the file containing airfoil polar data.

    Returns
    -------
    Callable[[float | np.ndarray], tuple[np.ndarray | float, np.ndarray | float]]
        Callable function accepting angle of attack in radians and returning
        interpolated $(C_l, C_d)$ values.

    Notes
    -----
    - Angles of attack in degrees are converted internally to radians.
    - Smoothing factors $s=0.1$ ($C_l$) and $s=0.001$ ($C_d$) are applied.
    """
    alpha_list: list[float] = []
    cl_list: list[float] = []
    cd_list: list[float] = []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Detect format based on header structure
    is_qblade = any(
        "alpha" in line.lower() for line in lines[:15]
    ) and not any("EOT" in line for line in lines)

    if is_qblade:
        data_lines = lines[11:]
    else:
        data_lines = []
        for line in lines[13:]:
            if "EOT" in line:
                break
            data_lines.append(line)

    # Parse numerical data lines
    for line in data_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            alpha_list.append(float(parts[0]))
            cl_list.append(float(parts[1]))
            cd_list.append(float(parts[2]))
        except ValueError:
            continue

    # Convert degrees to radians
    alpha_arr = np.radians(np.array(alpha_list, dtype=np.float64))
    cl_arr = np.array(cl_list, dtype=np.float64)
    cd_arr = np.array(cd_list, dtype=np.float64)

    # Create 1D spline interpolators
    afcl = UnivariateSpline(alpha_arr, cl_arr, s=0.1)
    afcd = UnivariateSpline(alpha_arr, cd_arr, s=0.001)

    def af(
        alpha: float | np.ndarray
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Evaluate interpolated lift ($C_l$) and drag ($C_d$) coefficients.

        Parameters
        ----------
        alpha : float or np.ndarray
            Angle of attack in radians.

        Returns
        -------
        cl : float or np.ndarray
            Interpolated lift coefficient(s).
        cd : float or np.ndarray
            Interpolated drag coefficient(s).
        """
        return afcl(alpha), afcd(alpha)

    return af

# ==============================================================================
# CONFIGURATION LOADER
# ==============================================================================

@lru_cache(maxsize=1)
def load_aero_config(path: str = "src/pyvawt/config/config.yaml") -> dict[str, Any]:
    """
    Load YAML configuration file into a dictionary with RAM caching.

    Parameters
    ----------
    path : str, default='src/pyvawt/config/config.yaml'
        Path to the target YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    ValueError
        If the file is empty or unparseable.
    KeyError
        If required top-level sections ('turbine', 'environment', 'solver') are missing.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file '{path}' is empty or invalid.")

    for section in ["turbine", "environment", "solver"]:
        if section not in config:
            raise KeyError(
                f"Required section '{section}' missing from configuration file."
            )

    def ensure_list(section_name: str, key_name: str) -> None:
        if key_name in config[section_name] and not isinstance(
            config[section_name][key_name], list
        ):
            config[section_name][key_name] = [config[section_name][key_name]]

    ensure_list("turbine", "chord")
    ensure_list("turbine", "solidity")
    ensure_list("environment", "Vinf")

    if "neuralfoil" in config["solver"] and "airfoil" in config["solver"]["neuralfoil"]:
        nf_cfg = config["solver"]["neuralfoil"]
        if not isinstance(nf_cfg["airfoil"], list):
            nf_cfg["airfoil"] = [nf_cfg["airfoil"]]

    return config


def get_cl_cd_neuralfoil(
    alpha: np.ndarray,
    W: float | np.ndarray,
    turbine_index: int,
    airfoil_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate lift ($C_l$) and drag ($C_d$) coefficients using NeuralFoil.

    Parameters
    ----------
    alpha : np.ndarray
        1D array of angles of attack in radians.
    W : float or np.ndarray
        Relative flow velocity in m/s.
    turbine_index : int
        Index referencing turbine parameters in the configuration.
    airfoil_index : int
        Index referencing the airfoil profile in the configuration.

    Returns
    -------
    cl : np.ndarray
        Evaluated lift coefficients matching input `alpha` shape.
    cd : np.ndarray
        Evaluated drag coefficients matching input `alpha` shape.
    """
    config = load_aero_config()

    chord = config["turbine"]["chord"][turbine_index]
    rho = config["environment"]["rho"]
    mu = config["environment"]["mu"]

    airfoil_name = config["solver"]["neuralfoil"]["airfoil"][airfoil_index]

    Re = rho * W * chord / mu
    mach = W / 343.2

    airfoil = asb.Airfoil(name=airfoil_name)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=np.rad2deg(alpha),
        Re=Re,
        mach=mach,
        model_size=config["solver"]["neuralfoil"]["model_size"],
        include_360_deg_effects=False,
    )

    cl = np.asarray(aero["CL"], dtype=np.float64).reshape(alpha.shape)
    cd = np.asarray(aero["CD"], dtype=np.float64).reshape(alpha.shape)

    return cl, cd


# ==============================================================================
# AERODYNAMIC MODEL CLASSES
# ==============================================================================

class Aerodynamics:
    '''
    Abstract base class for aerodynamics models.
    '''
    def get_cl_cd(self, alpha, W=None):
        '''
        Returns the lift and drag coefficients.

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].
        W : float, optional
            Relative wind speed [m/s].

        Returns
        -------
        tuple of floats
            (Cl, Cd)
        '''
        raise NotImplementedError("Subclasses must implement this method.")


class NeuralFoilAerodynamics(Aerodynamics):
    """
    Aerodynamic model optimized with Look-Up Table (LUT) generation and RAM caching.

    Evaluates aerodynamic lift (Cl) and drag (Cd) coefficients using NeuralFoil
    pre-computed matrices and fast bilinear 2D interpolation.

    Parameters
    ----------
    turbine_index : int
        Index referencing turbine parameters in the configuration.
    airfoil_index : int
        Index referencing the airfoil profile in the configuration.
    config : dict, optional
        Simulation configuration dictionary. If None, attempts default path loading.
    n_alpha : int, default=721
                Number of grid discretization points for angle of attack [-π, π].
    n_W : int, default=50
        Number of grid discretization points for relative velocity W.
    W_min : float, default=0.1
        Minimum relative velocity boundary [m/s].
    W_max : float, default=150.0
        Maximum relative velocity boundary [m/s].

    Attributes
    ----------
    turbine_index : int
        Index referencing turbine parameters.
    airfoil_index : int
        Index referencing airfoil profile.
    W_min : float
        Minimum velocity boundary [m/s].
    W_max : float
        Maximum velocity boundary [m/s].
    alpha_grid : np.ndarray
        1D array of angle of attack grid points [rad].
    W_grid : np.ndarray
        1D array of relative velocity grid points [m/s].
    cl_table : np.ndarray
        2D matrix of precomputed lift coefficients [-].
    cd_table : np.ndarray
        2D matrix of precomputed drag coefficients [-].
    """

    def __init__(
        self,
        turbine_index: int,
        airfoil_index: int,
        config: dict | None = None,
        n_alpha: int = 721,
        n_W: int = 50,
        W_min: float = 0.1,
        W_max: float = 150.0,
    ) -> None:
        self.turbine_index = turbine_index
        self.airfoil_index = airfoil_index
        self.W_min = W_min
        self.W_max = W_max

        if config is None:
            try:
                config = load_aero_config()
            except TypeError:
                config = load_aero_config("src/pyvawt/config/config.yaml")

        airfoil_cfg = config["solver"]["neuralfoil"]["airfoil"]
        airfoil_name = (
            airfoil_cfg[airfoil_index]
            if isinstance(airfoil_cfg, (list, tuple))
            else airfoil_cfg
        )

        chord_cfg = config["turbine"]["chord"]
        chord = (
            chord_cfg[turbine_index]
            if isinstance(chord_cfg, (list, tuple))
            else chord_cfg
        )
        rho = config["environment"]["rho"]
        mu = config["environment"]["mu"]

        cache_key = (airfoil_name, chord, rho, mu, n_alpha, n_W, W_min, W_max)

        # Check if the table is already cached in RAM
        if cache_key in _AERO_LUT_CACHE:
            (
                self.alpha_grid,
                self.W_grid,
                self.cl_table,
                self.cd_table,
            ) = _AERO_LUT_CACHE[cache_key]
        else:
            # Compute and cache matrices if evaluated for the first time
            self.alpha_grid = np.linspace(-np.pi, np.pi, n_alpha)
            self.W_grid = np.linspace(W_min, W_max, n_W)

            self.cl_table = np.zeros((n_alpha, n_W), dtype=np.float64)
            self.cd_table = np.zeros((n_alpha, n_W), dtype=np.float64)

            for j, W_val in enumerate(self.W_grid):
                cl_vec, cd_vec = get_cl_cd_neuralfoil(
                    self.alpha_grid, W_val, turbine_index, airfoil_index
                )
                self.cl_table[:, j] = cl_vec
                self.cd_table[:, j] = cd_vec

            _AERO_LUT_CACHE[cache_key] = (
                self.alpha_grid,
                self.W_grid,
                self.cl_table,
                self.cd_table,
            )

    def get_cl_cd(
        self, alpha: float | np.ndarray, W: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """
        Evaluate lift (Cl) and drag (Cd) coefficients via 2D LUT interpolation.

        Parameters
        ----------
        alpha : float or np.ndarray
            Angle of attack [rad].
        W : float or np.ndarray
            Relative flow velocity [m/s].

        Returns
        -------
        cl : float or np.ndarray
            Interpolated lift coefficient(s) [-].
        cd : float or np.ndarray
            Interpolated drag coefficient(s) [-].
        """
        # Ensure array format alignment and convert to 1D contiguous arrays
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
        W_arr = np.atleast_1d(np.asarray(W, dtype=np.float64))

        alpha_b, W_b = np.broadcast_arrays(alpha_arr, W_arr)
        alpha_flat = alpha_b.ravel()
        W_flat = W_b.ravel()

        # Execute Numba kernel interpolation passing 5 required parameters
        cl_flat = interpolate_2d_lut(
            alpha_flat, W_flat, self.alpha_grid, self.W_grid, self.cl_table
        )
        cd_flat = interpolate_2d_lut(
            alpha_flat, W_flat, self.alpha_grid, self.W_grid, self.cd_table
        )

        # Return scalars or arrays matching original input structure
        if np.isscalar(alpha) and np.isscalar(W):
            return cl_flat[0], cd_flat[0]

        return cl_flat.reshape(alpha_b.shape), cd_flat.reshape(alpha_b.shape)

class FileAerodynamics(Aerodynamics):
    '''
    Aerodynamics model using airfoil data from a file to calculate Cl and Cd.

    Parameters
    ----------
    filename : str
        Path to the file containing airfoil data.
    '''
    def __init__(self, filename):
        self.af_func = readaerodyn(filename)

    def get_cl_cd(self, alpha, W=None):
        '''
        Returns Cl and Cd interpolated from airfoil data.

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].
        W : float, optional
            Not used, kept for compatibility with interface.

        Returns
        -------
        tuple of floats
            (Cl, Cd)
        '''
        return self.af_func(alpha)


# =============================
# AERODYNAMICS HELPERS
# =============================

def mach(W: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate Mach number assuming standard atmospheric speed of sound (343.2 m/s).
    """
    return W / 343.2


def get_tc_from_airfoil(airfoil_name: str) -> float:
    """
    Extract thickness-to-chord ratio (t/c) from NACA 4-digit airfoil designations.

    Parameters
    ----------
    airfoil_name : str
        Name of the airfoil (e.g. 'naca0012', 'NACA 0018').

    Returns
    -------
    float
        Thickness-to-chord ratio.
    """
    clean_name = airfoil_name.lower().replace(" ", "")

    if clean_name.startswith("naca") and len(clean_name) == 8:
        thickness_digits = clean_name[-2:]
        return int(thickness_digits) / 100.0

    raise ValueError(
        f"Unable to determine t/c ratio for airfoil: '{airfoil_name}'"
    )


def detect_stall_angles(
    config: dict[str, Any], airfoil_index: int
) -> tuple[float, float]:
    """
    Compute positive and negative stall angles using NeuralFoil and AeroSandbox.

    Parameters
    ----------
    config : dict
        Simulation configuration dictionary.
    airfoil_index : int
        Index of the target airfoil profile in configuration.

    Returns
    -------
    aoaStallPos : float
        Positive stall angle of attack [rad].
    aoaStallNeg : float
        Negative stall angle of attack [rad].
    """
    import aerosandbox as asb
    import neuralfoil as nf

    airfoil_name = config["solver"]["neuralfoil"]["airfoil"][
        airfoil_index
    ].lower()
    airfoil = asb.Airfoil(airfoil_name)

    alpha_deg = np.linspace(-30, 30, 600)

    # Reference flow conditions
    Re = 2.5e6
    model_size = "large"

    aero = nf.get_aero_from_airfoil(
        airfoil=airfoil, alpha=alpha_deg, Re=Re, model_size=model_size
    )

    cl = aero["CL"]

    idx_pos = int(np.argmax(cl))
    idx_neg = int(np.argmin(cl))

    aoaStallPos = float(np.deg2rad(alpha_deg[idx_pos]))
    aoaStallNeg = float(np.deg2rad(alpha_deg[idx_neg]))

    if not np.isclose(aoaStallPos, -aoaStallNeg, rtol=1e-3):
        UI.status(
            "Airfoil Check",
            f"Asymmetric stall detected for '{airfoil_name}': "
            f"+{np.degrees(aoaStallPos):.1f}° / {np.degrees(aoaStallNeg):.1f}°",
            level="warn",
        )

    return aoaStallPos, aoaStallNeg


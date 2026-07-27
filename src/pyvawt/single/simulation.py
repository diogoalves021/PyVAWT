import numba as nb
import numpy as np
import math
import os
import h5py
import csv
import time
from datetime import timedelta
import traceback
import copy
from scipy.integrate import quad
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from typing import Any

from src.pyvawt.submodels.flow_curvature import FlowCurvatureManager, FlowCurvatureModel
#from src.pyvawt.submodels.boeing_vertol import af, Boeing_Vertol
from src.pyvawt.submodels.boeing_vertol import boeing_vertol_jit, interp2d_scalar
from src.pyvawt.single.data_reading import readaerodyn
from src.pyvawt.single.utils import save_config, format_time
from src.pyvawt.single.data_generation import get_cl_cd_neuralfoil
from src.pyvawt.single.utils import load_config, get_tc_from_airfoil, detect_stall_angles, save_config, resolve_turbine_geometry, UI

# Coefficients of influence
@nb.njit(fastmath=True, cache=True)
def Dxintegrand(x, y, phi):
    '''
    Integrand function for computing Dx.
    '''
    v1 = x + np.sin(phi)
    v2 = y - np.cos(phi)
    # Nota: O print(v1, v2) foi removido pois causava travamento extremo de I/O em loops
    return (v1 * np.sin(phi) - v2 * np.cos(phi)) / (2.0 * np.pi * (v1 * v1 + v2 * v2))

@nb.njit(fastmath=True, cache=True)
def Ayintegrand(x, y, phi):
    '''
    Integrand function for computing Ay.
    '''
    v1 = x + np.sin(phi)
    v2 = y - np.cos(phi)
    if abs(v1) < 1e-12 and abs(v2) < 1e-12:
        return 0.0
    return (v1 * np.cos(phi) + v2 * np.sin(phi)) / (2.0 * np.pi * (v1 * v1 + v2 * v2))

@nb.njit(fastmath=True, cache=True)
def panelIntegration(xvec, yvec, thetavec, ifunc, n_substeps=16):
    '''
    Perform panel integration to compute influence coefficients using 
    Composite Trapezoidal rule compiled in C via Numba.
    '''
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0]
    A = np.zeros((nx, ntheta), dtype=np.float64)

    for i in range(nx):
        x = xvec[i]
        y = yvec[i]
        for j in range(ntheta):
            # Limites de integração do painel j
            a = thetavec[j] - dtheta / 2.0
            b = thetavec[j] + dtheta / 2.0
            h = (b - a) / n_substeps
            
            # Regra trapezoidal composta sobre 'ifunc'
            sum_val = 0.5 * (ifunc(x, y, a) + ifunc(x, y, b))
            for k in range(1, n_substeps):
                sum_val += ifunc(x, y, a + k * h)
                
            A[i, j] = sum_val * h

    return A

@nb.njit(fastmath=True, cache=True)
def AyIJ(xvec, yvec, thetavec):
    '''
    Compute AyIJ by integrating with the Ayintegrand function.
    '''
    return panelIntegration(xvec, yvec, thetavec, Ayintegrand)

@nb.njit(fastmath=True, cache=True)
def DxIJ(xvec, yvec, thetavec):
    '''
    Compute DxIJ by integrating with the Dxintegrand function.
    '''
    return panelIntegration(xvec, yvec, thetavec, Dxintegrand)

def WxIJ(xvec, yvec, thetavec):
    '''
    Compute WxIJ by processing the x and y coordinates to determine influence of panels.

    This function initializes a Wx matrix based on the given x and y coordinates 
    and the angles at which the integration occurs.

    Parameters
    ----------
    xvec : ndarray
        Array of x-coordinates of the panels.
        
    yvec : ndarray
        Array of y-coordinates of the panels.
        
    thetavec : ndarray
        Array of angles (in radians) at which the integration is performed.
        
    Returns
    -------
    Wx : ndarray
        The Wx matrix resulting from the panel influence calculations.
    '''
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] # Assumes equally spaced values
    Wx = np.zeros((nx, ntheta))

    for i in range(nx):
        if (
            -1.0 <= yvec[i] <= 1.0
            and xvec[i] >= 0.0
            and xvec[i]**2 + yvec[i]**2 >= 1.0
        ):
            thetak = np.arccos(yvec[i])
            k = np.searchsorted(thetavec + dtheta / 2, thetak, side='right')
            if 0 <= k < ntheta:
                Wx[i, k] = -1.0
                Wx[i, ntheta - k - 1] = 1.0

    return Wx
@nb.jit
def DxII(thetavec):
    '''
    Compute DxII based on the given angles.

    This function initializes a matrix Rx for influence calculations based on 
    the angular distribution in `thetavec`.

    Parameters
    ----------
    thetavec : ndarray
        Array of angles (in radians) at which the integration is performed.
        
    Returns
    -------
    Rx : ndarray
        The DxII matrix resulting from the calculations based on the angles.
    '''
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] # Assumes equally spaced values
    Rx = (dtheta / (4 * np.pi)) * np.ones((ntheta, ntheta))

    for i in range(ntheta):
        if i < ntheta // 2:
            Rx[i, i] = (-1 + 1.0 / ntheta) / 2.0
        else:
            Rx[i, i] = (1 + 1.0 / ntheta) / 2.0

    return Rx
@nb.jit
def WxII(thetavec):
    '''
    Generate the Wx matrix for a given set of angular divisions.

    Parameters
    ----------
    thetavec : array_like
        Array of angular values (theta) used to construct the Wx matrix.

    Returns
    -------
    Wx : ndarray
        A square matrix of size (ntheta, ntheta) representing the influence 
        of Wx between different turbines.

    Notes
    -----
    The function generates the Wx matrix for a set of angles, where half of the 
    matrix is filled with `-1` values and the other half with `1` values.
    '''
    ntheta = len(thetavec)
    Wx = np.zeros((ntheta, ntheta))

    for i in range(ntheta // 2, ntheta):
        Wx[i, ntheta - 1 - i] = -1

    return Wx

def precomputeMatrices(ntheta, modulepath):
    '''
    Precompute matrices for self-influence and save them in an HDF5 file.

    Parameters
    ----------
    ntheta : int
        Number of angular divisions (theta).
    modulepath : str
        Path to the directory where the HDF5 file will be saved.

    Returns
    -------
    filepath : str
        Path to the saved HDF5 file containing the precomputed matrices.

    Notes
    -----
    The function computes the self-influence matrices for Dx, Wx, and Ay, 
    then stores them in an HDF5 file for future use.
    '''
    dtheta = 2 * np.pi / ntheta
    theta = np.arange(dtheta / 2, 2 * np.pi, dtheta)

    # Precompute matrices
    Dxself = DxII(theta)
    Wxself = WxII(theta)
    Ayself = AyIJ(-np.sin(theta), np.cos(theta), theta)

    filepath = f'{modulepath}/theta-{ntheta}.h5'
    with h5py.File(filepath, 'w') as file:
        file.create_dataset('theta', data=theta)
        file.create_dataset('Dx', data=Dxself)
        file.create_dataset('Wx', data=Wxself)
        file.create_dataset('Ay', data=Ayself)

    return filepath

def matrixAssemble(centerX, centerY, radius, ntheta):
    '''
    Assemble the influence matrices for a single VAWT turbine based on 
    its center coordinates, radius, and number of angular discretization points.

    Parameters
    ----------
    centerX : float
        X-coordinate of the turbine center.
    centerY : float
        Y-coordinate of the turbine center.
    radius : float
        Radius of the turbine.
    ntheta : int
        Number of angular divisions (discretization steps along the turbine's perimeter).

    Returns
    -------
    Ax : ndarray of shape (ntheta, ntheta)
        Assembled matrix combining self-induction and wake effects in the x-direction.
    Ay : ndarray of shape (ntheta, ntheta)
        Assembled matrix capturing self-induction effects in the y-direction.
    theta : ndarray of shape (ntheta,)
        Array of angular positions along the turbine’s circular path (in radians).

    Notes
    -----
    This function is tailored for simulations involving a single vertical-axis wind turbine (VAWT).
    It loads precomputed influence matrices (Dx, Wx, Ay) for a turbine with the given angular resolution,
    avoiding the need for pairwise interaction calculations present in multi-turbine simulations.
    '''
    file = f'theta-{ntheta}.h5'
    modulepath = os.getcwd() # uses the current directory as the path
    if not os.path.isfile(file):
        filepath = precomputeMatrices(ntheta, modulepath)
    else:
        filepath = os.path.join(modulepath, file)

    # Load precomputed matrices from the HDF5 file
    with h5py.File(filepath, 'r') as f:
        theta = f['theta'][:]
        Dxself = f['Dx'][:]
        Wxself = f['Wx'][:]
        Ayself = f['Ay'][:]

    # For a single turbine, all the global matrice is a single self set of parameters
    Dx = Dxself.copy()
    Wx = Wxself.copy()
    Ay = Ayself.copy()

    # Calculate Ax matrix
    Ax = Dx + Wx

    return Ax, Ay, theta

#---------------------------------------
#
#-------- Force coeffients --------

class Turbine:
    '''
    Class representing a vertical-axis wind turbine (VAWT).

    Parameters
    ----------
    r : float
        Turbine radius [m].
    chord : float
        Blade chord length [m].
    twist : float
        Blade pitch angle [rad].
    delta : float
        Tilt angle of the turbine [rad].
    B : int
        Number of blades.
    Omega : float
        Rotational speed of the turbine [rad/s].
    '''
    def __init__(self, r: float, chord: float,
                twist: float, delta: float,
                B: int, Omega: float,
                centerX: float, centerY: float,
                 solidity:float, aero_model = None):
        self.r = r
        self.chord = chord
        self.twist = twist
        self.delta = delta
        self.B = B
        self.Omega = Omega
        self.centerX = centerX
        self.centerY = centerY
        self.solidity = solidity
        self.aero = aero_model

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

@nb.njit(fastmath=True, cache=True)
def interpolate_2d_lut(
    alpha_vec: np.ndarray,
    W_vec: np.ndarray,
    alpha_grid: np.ndarray,
    W_grid: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:
    """
    Perform fast 2D bilinear interpolation over a structured Lookup Table (LUT).

    Maps query angles of attack to the interval [-π, π] and evaluates table values
    using index bounding and uniform grid spacing.

    Parameters
    ----------
    alpha_vec : np.ndarray
        1D array of query angle of attack values [rad].
    W_vec : np.ndarray
        1D array of query relative velocity values [m/s].
    alpha_grid : np.ndarray
        1D array of uniformly spaced grid coordinates for angle of attack [rad].
    W_grid : np.ndarray
        1D array of uniformly spaced grid coordinates for relative velocity [m/s].
    table : np.ndarray
        2D matrix of shape `(len(alpha_grid), len(W_grid))` containing the
        precomputed values to interpolate (e.g., aerodynamic coefficients).

    Returns
    -------
    np.ndarray
        1D array of bilinearly interpolated values corresponding to each (`alpha_vec`, `W_vec`) pair.
    """
    n = len(alpha_vec)
    out = np.empty(n, dtype=np.float64)

    n_alpha = len(alpha_grid)
    n_w = len(W_grid)

    d_alpha = alpha_grid[1] - alpha_grid[0]
    d_w = W_grid[1] - W_grid[0]

    alpha_min, alpha_max = alpha_grid[0], alpha_grid[-1]
    w_min, w_max = W_grid[0], W_grid[-1]

    for i in range(n):
        # Map angle of attack to [-pi, pi]
        a = (alpha_vec[i] + np.pi) % (2.0 * np.pi) - np.pi
        w = W_vec[i]

        # Index search and bounding for alpha
        if a <= alpha_min:
            ia = 0
            u = 0.0
        elif a >= alpha_max:
            ia = n_alpha - 2
            u = 1.0
        else:
            pos_a = (a - alpha_min) / d_alpha
            ia = int(pos_a)
            u = pos_a - ia

        # Index search and bounding for relative velocity W
        if w <= w_min:
            iw = 0
            v = 0.0
        elif w >= w_max:
            iw = n_w - 2
            v = 1.0
        else:
            pos_w = (w - w_min) / d_w
            iw = int(pos_w)
            v = pos_w - iw

        # Bilinear interpolation
        f00 = table[ia, iw]
        f10 = table[ia + 1, iw]
        f01 = table[ia, iw + 1]
        f11 = table[ia + 1, iw + 1]

        out[i] = (
            (1.0 - u) * (1.0 - v) * f00
            + u * (1.0 - v) * f10
            + (1.0 - u) * v * f01
            + u * v * f11
        )

    return out


# Global cache dictionary for storing Aerodynamic Look-Up Tables (LUT) in RAM
_AERO_LUT_CACHE: dict = {}

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
                config = load_config()
            except TypeError:
                config = load_config("src/pyvawt/config/config.yaml")

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
        

class Environment:
    '''
    Class representing the wind and fluid properties of the environment.

    Parameters
    ----------
    Vinf : float
        Free stream wind velocity [m/s].
    rho : float
        Air density [kg/m^3].
    mu : float
        Dynamic viscosity of the air [Pa·s].
    '''
    def __init__(self, Vinf: float, rho: float, mu: float):
        self.Vinf = Vinf
        self.rho = rho
        self.mu = mu
        self.BV_DynamicFlagL = 0
        self.BV_DynamicFlagD = 0


@nb.njit(fastmath=True, cache=True)
def fast_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute 1D numerical integration using the trapezoidal rule.

    Parameters
    ----------
    y : np.ndarray
        1D array of function values to integrate.
    x : np.ndarray
        1D array of sample points corresponding to `y`.

    Returns
    -------
    float
        Approximated integral value.
    """
    n = len(x)
    integral = 0.0
    for i in range(n - 1):
        integral += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return integral


def warmup_numba_kernels(verbose: bool = True) -> None:
    """
    Pre-compile or load cached Numba JIT kernels prior to simulation execution.

    Parameters
    ----------
    verbose : bool, default=True
        If True, displays initialization status and compilation elapsed time
        via the UI helper.
    """
    if verbose:
        UI.status("JIT Engine (Numba)", "Compiling C kernels...", level="info")

    t0 = time.perf_counter()
    dummy_1d = np.zeros(10, dtype=np.float64)
    dummy_grid = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
    dummy_table = np.zeros((10, 10), dtype=np.float64)

    try:
        _radialforce_kernel(
            dummy_1d, dummy_1d, dummy_grid, 10.0, 1.0, 0.0, 0.0, 3, 10.0, 0.1,
            10.0, 1.2, 1.8e-5, dummy_grid, dummy_grid, dummy_table, dummy_table,
            True, 0.2, -0.2, 0.0, 0.12
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

@nb.njit(fastmath=True, cache=True)
def _radialforce_kernel(
    uvec: np.ndarray,
    vvec: np.ndarray,
    thetavec: np.ndarray,
    r: float,
    chord: float,
    twist: float,
    delta: float,
    B: int,
    Omega: float,
    solidity: float,
    Vinf: float,
    rho: float,
    mu: float,
    alpha_grid: np.ndarray,
    W_grid: np.ndarray,
    cl_table: np.ndarray,
    cd_table: np.ndarray,
    use_dynamic_stall: bool,
    aoa_stall_pos: float,
    aoa_stall_neg: float,
    AOA0: float,
    tc: float,
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Execute high-density aerodynamic kernel for VAWT blade element calculations.

    Computes velocity triangles, static/dynamic lift and drag coefficients, 
    instantaneous force decompositions, integrated thrust/power coefficients,
    and induction correction factors over an azimuthal revolution.

    Parameters
    ----------
    uvec : np.ndarray
        1D array of axial induction factor distribution along azimuth [-].
    vvec : np.ndarray
        1D array of tangential induction factor distribution along azimuth [-].
    thetavec : np.ndarray
        1D array of azimuthal angles [rad].
    r : float
        Rotor radius [m].
    chord : float
        Blade chord length [m].
    twist : float
        Blade pitch/twist angle [rad].
    delta : float
        Cone/blade inclination angle [rad].
    B : int
        Number of blades [-].
    Omega : float
        Rotor angular rotational speed [rad/s].
    solidity : float
        Target rotor solidity [-].
    Vinf : float
        Freestream wind speed [m/s].
    rho : float
        Air density [kg/m³].
    mu : float
        Dynamic air viscosity [Pa·s].
    alpha_grid : np.ndarray
        1D grid array for angle of attack LUT lookup [rad].
    W_grid : np.ndarray
        1D grid array for relative velocity LUT lookup [m/s].
    cl_table : np.ndarray
        2D matrix of precomputed static lift coefficients [-].
    cd_table : np.ndarray
        2D matrix of precomputed static drag coefficients [-].
    use_dynamic_stall : bool
        Flag indicating whether to apply the Boeing-Vertol dynamic stall model.
    aoa_stall_pos : float
        Positive static stall angle of attack [rad].
    aoa_stall_neg : float
        Negative static stall angle of attack [rad].
    AOA0 : float
        Zero-lift angle of attack [rad].
    tc : float
        Airfoil relative thickness-to-chord ratio [-].

    Returns
    -------
    q : np.ndarray
        Azimuthal distribution of normal force loading parameter [-].
    ka : float
        Induction correction factor [-].
    CT : float
        Integrated rotor thrust coefficient [-].
    CP : float
        Integrated rotor power coefficient [-].
    Rp : np.ndarray
        Normal/radial force per unit span distribution along azimuth [N/m].
    Tp : np.ndarray
        Tangential force per unit span distribution along azimuth [N/m].
    Zp : np.ndarray
        Vertical/spanwise force per unit span distribution along azimuth [N/m].
    alpha : np.ndarray
        Distribution of local angle of attack along azimuth [rad].
    W : np.ndarray
        Distribution of local relative flow velocity along azimuth [m/s].
    """
    n = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] if n > 1 else 0.0
    rotation = 1.0 if Omega >= 0 else -1.0
    abs_Omega = abs(Omega)

    # Velocity Triangle and Angle of Attack
    Vn = Vinf * (1.0 + uvec) * np.sin(thetavec) - Vinf * vvec * np.cos(thetavec)
    Vt = rotation * (
        Vinf * (1.0 + uvec) * np.cos(thetavec) + Vinf * vvec * np.sin(thetavec)
    ) + abs_Omega * r

    W = np.sqrt(Vn**2 + Vt**2)
    phi = np.arctan2(Vn, Vt)
    alpha = phi - twist

    # Static Coefficient Interpolation via LUT
    cl_stat = np.empty(n, dtype=np.float64)
    cd_stat = np.empty(n, dtype=np.float64)
    cm_stat = np.zeros(n, dtype=np.float64)

    for i in range(n):
        cl_stat[i] = interp2d_scalar(alpha[i], W[i], alpha_grid, W_grid, cl_table)
        cd_stat[i] = interp2d_scalar(alpha[i], W[i], alpha_grid, W_grid, cd_table)

    # Dynamic Stall Submodel (Boeing-Vertol)
    cl = cl_stat.copy()
    cd = cd_stat.copy()
    cm = cm_stat.copy()

    if use_dynamic_stall:
        dt = dtheta / abs_Omega if abs_Omega > 1e-6 else 1.0
        flagL = 0
        flagD = 0

        for i in range(n):
            # Time derivative of angle of attack d(alpha)/dt
            i_prev = n - 1 if i == 0 else i - 1
            dalpha_dt = (alpha[i] - alpha[i_prev]) / dt
            w_val = W[i] if W[i] > 1e-6 else 1e-6
            adotnorm = (dalpha_dt * chord) / (2.0 * w_val)
            umach = w_val / 343.0

            # Direct Boeing-Vertol JIT kernel execution
            c_l, c_d, c_m, flagL, flagD = boeing_vertol_jit(
                cl_stat[i],
                cd_stat[i],
                cm_stat[i],
                alpha[i],
                adotnorm,
                umach,
                w_val,
                aoa_stall_pos,
                aoa_stall_neg,
                AOA0,
                tc,
                flagL,
                flagD,
                alpha_grid,
                W_grid,
                cl_table,
                cd_table,
            )
            cl[i] = c_l
            cd[i] = c_d
            cm[i] = c_m

    # Normal/Tangential Coefficients and Solidity
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cn = cl * cos_phi + cd * sin_phi
    ct = cl * sin_phi - cd * cos_phi

    sigma_r = B * chord / r
    sigma_D = B * chord / (2.0 * r)
    sigma = (
        sigma_r
        if abs(solidity - sigma_r) < abs(solidity - sigma_D)
        else 2.0 * solidity
    )

    q = sigma / (4.0 * np.pi) * cn * (W / Vinf) ** 2

    # Instantaneous Force Decomposition
    qdyn = 0.5 * rho * W**2
    cos_delta = np.cos(delta)
    tan_delta = np.tan(delta)

    Rp = -cn * qdyn * chord
    Tp = ct * qdyn * chord / cos_delta
    Zp = -cn * qdyn * chord * tan_delta

    # Azimuthal Integration for Thrust Coefficient (CT) and Induction Factor (ka)
    integrand = (W / Vinf) ** 2 * (
        cn * np.sin(thetavec) - rotation * ct * np.cos(thetavec) / cos_delta
    )
    CT = sigma / (4.0 * np.pi) * fast_trapz(integrand, thetavec)

    if CT > 2.0:
        a = 0.5 * (1.0 + np.sqrt(1.0 + CT))
        ka = 1.0 / (a - 1.0)
    elif CT > 0.96:
        a = (1.0 / 7.0) * (1.0 + 3.0 * np.sqrt(3.5 * CT - 3.0))
        ka = 18.0 * a / (7.0 * a**2 - 2.0 * a + 4.0)
    else:
        a = 0.5 * (1.0 - np.sqrt(1.0 - CT))
        ka = 1.0 / (1.0 - a)

    # Integrated Power Coefficient (CP)
    Q = r * Tp
    P = abs_Omega * B / (2.0 * np.pi) * fast_trapz(Q, thetavec)
    CP = P / (0.5 * rho * (Vinf**3) * (2.0 * r))

    return q, ka, CT, CP, Rp, Tp, Zp, alpha, W

def radialforce(
    uvec: Any,
    vvec: Any,
    thetavec: Any,
    turbine: Any,
    env: Any,
    *args: Any,
    use_dynamic_stall: bool = True,
    **kwargs: Any,
) -> tuple[
    np.ndarray,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Flexible Python wrapper for the JIT-compiled radial force Numba kernel.

    Unpacks turbine geometric attributes, fluid environment parameters, and 
    aerodynamic LUT objects, casting them into C-compatible primitives prior to 
    dispatching execution to `_radialforce_kernel`. Variadic arguments (`*args` 
    and `**kwargs`) ensure interface compatibility with 3D multi-slice solvers.

    Parameters
    ----------
    uvec : array_like
        1D distribution of axial induction factors along the azimuth [-].
    vvec : array_like
        1D distribution of tangential induction factors along the azimuth [-].
    thetavec : array_like
        1D distribution of azimuthal discretization angles [rad].
    turbine : object
        Turbine instance containing rotor geometry (`r`, `chord`, `twist`, `delta`, 
        `B`, `Omega`, `solidity`) and the aerodynamic model object (`aero`).
    env : object
        Environment instance containing fluid properties (`Vinf`, `rho`, `mu`).
    *args : tuple
        Optional positional arguments passed for 3D slice solver compatibility
        (e.g., `config`, `turbine_index`, `airfoil_index`, `flow_manager`, `z`, `H`).
    use_dynamic_stall : bool, default=True
        Flag to enable or disable the Boeing-Vertol dynamic stall model.
    **kwargs : dict
        Optional keyword arguments passed for extended solver compatibility.

    Returns
    -------
    tuple of (np.ndarray or float)
        Returns the output tuple from `_radialforce_kernel`:
        - `q` : Azimuthal distribution of normal force loading parameter [-].
        - `ka` : Induction correction factor [-].
        - `CT` : Integrated rotor thrust coefficient [-].
        - `CP` : Integrated rotor power coefficient [-].
        - `Rp` : Normal/radial force per unit span distribution [N/m].
        - `Tp` : Tangential force per unit span distribution [N/m].
        - `Zp` : Vertical/spanwise force per unit span distribution [N/m].
        - `alpha` : Local angle of attack distribution [rad].
        - `W` : Local relative flow velocity distribution [m/s].
    """
    aero = turbine.aero

    # Extract aerodynamic stall parameters with safety fallbacks
    aoa_stall_pos = float(getattr(aero, "aoa_stall_pos", np.radians(12.0)))
    aoa_stall_neg = float(getattr(aero, "aoa_stall_neg", np.radians(-12.0)))
    AOA0 = float(getattr(aero, "AOA0", 0.0))
    tc = float(getattr(aero, "tc", 0.12))

    # Dispatch to C-compiled Numba kernel
    return _radialforce_kernel(
        np.asarray(uvec, dtype=np.float64),
        np.asarray(vvec, dtype=np.float64),
        np.asarray(thetavec, dtype=np.float64),
        float(turbine.r),
        float(turbine.chord),
        float(turbine.twist),
        float(turbine.delta),
        int(turbine.B),
        float(turbine.Omega),
        float(turbine.solidity),
        float(env.Vinf),
        float(env.rho),
        float(env.mu),
        aero.alpha_grid,
        aero.W_grid,
        aero.cl_table,
        aero.cd_table,
        bool(use_dynamic_stall),
        aoa_stall_pos,
        aoa_stall_neg,
        AOA0,
        tc,
    )
'''
def apply_tip_loss(cn, ct, z, H, turbine, env, Vinf, uvec):
    B = turbine.B
    Omega = abs(turbine.Omega)

    if Omega < 1e-6:
        return cn, ct
    
    # Omega_ref = 5.8 * Vinf / turbine.r # The constant is the optimal TSR
    Omega_ref = Omega
    Ve = Vinf * (1 + np.mean(uvec))  # change this to the velocity between disks ; MUDEI PARA APENAS VINF PARA TESTAR E FOI A MELHOR CURVA ATE AGORA
    s = np.pi * Ve / (B * Omega_ref)

    g = np.pi * (H/2 - abs(z)) / s

    val = np.exp(-g)
    val = np.clip(val, 0.0, 1.0)

    F = (2/np.pi) * np.arccos(val)
    F = np.clip(F, 0.0, 1.0)

    return F * cn, F * ct, F
# Function under testing
'''
def apply_tip_loss(cn, ct, z, H, turbine, env, Vinf, uvec,
                   lambda_ref=5.8, alpha=0.1, Fmin=0.87, g_scale=0.35):
    B = turbine.B
    Omega = abs(turbine.Omega)
    Vinf = max(Vinf, 1e-12)

    # velocidade convectiva simples e estável
    Ve = Vinf

    # TSR operacional
    lam = Omega * turbine.r / Vinf

    # TSR efetivo suavizado:
    # evita que o tip-loss fique brutal em TSR baixo
    lam_eff = alpha * lam + (1.0 - alpha) * lambda_ref

    Omega_eff = lam_eff * Vinf / turbine.r

    s = np.pi * Ve / (B * Omega_eff)

    span_dist = max(H / 2 - abs(z), 0.0)
    g = g_scale*np.pi*span_dist/s 

    val = np.exp(-g)
    val = np.clip(val, 0.0, 1.0)

    F = (2 / np.pi) * np.arccos(val)
    F = np.clip(F, Fmin, 1.0)

    return F * cn, F * ct, F
#------------------------------------
#
#-------- solve the system --------

def residual(w, A, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager, z=None, H=None):
    '''
    Compute the residual for the actuator-cylinder equations of a single VAWT.

    Parameters
    ----------
    w : ndarray of shape (2*ntheta,)
        Solution vector [u_0, ..., u_{ntheta-1}, v_0, ..., v_{ntheta-1}].
    A : ndarray of shape (2*ntheta, ntheta)
        Combined influence matrix: [Ax; Ay].
    theta : ndarray of shape (ntheta,)
        Angular discretization points along the turbine perimeter.
    turbine : Turbine
        Single turbine object containing geometry and performance data.
    env : Environment
        Environmental conditions (wind speed, density, viscosity, etc.).
    config : dict or Config
        Additional configuration parameters for the simulation.
    turbine_index : int
        Index of this turbine in a larger fleet (for lookup of data tables).
    airfoil_index : int
        Index of the airfoil used on this turbine.

    Returns
    -------
    ndarray of shape (2*ntheta,)
        Residual vector: (A @ q) * k_mult - w.
    '''
    # Initial configuration
    ntheta = len(theta)

    # split u and v
    u = w[:ntheta]
    v = w[ntheta:]

    # Compute radial force, returns q (length ntheta) and the scalar kappa (ka)
    q, ka, *_ = radialforce(u, v, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager, z, H)

    # Build k_mult twice (for u- and v-equations)
    kmult = np.full(2 * ntheta, ka)

    return (A @ q) * kmult - w

def actuatorcylinder(turbine, env, ntheta, config, turbine_index, airfoil_index, flow_manager, z=None, H=None):
    '''
    Solve the actuator-cylinder model for a single VAWT turbine.

    Parameters
    ----------
    turbine : Turbine
        The turbine object to be used in the simulation.
    env : Environment
        Environmental conditions (wind speed, density, viscosity, etc.).
    ntheta : int
        Number of angular discretization points along the turbine perimeter.
    config : dict or Config
        Additional configuration parameters for the simulation.
    turbine_index : int
        Index of this turbine in a larger fleet (for data lookup).
    airfoil_index : int
        Index of the airfoil used on this turbine.

    Returns
    -------
    CT : float
        Thrust coefficient of the turbine.
    CP : float
        Power coefficient of the turbine.
    Rp : ndarray of shape (ntheta,)
        Radial force distribution around the turbine.
    Tp : ndarray of shape (ntheta,)
        Tangential force distribution around the turbine.
    Zp : ndarray of shape (ntheta,)
        Radial positions corresponding to the computed forces.
    theta : ndarray of shape (ntheta,)
        Angular discretization points (radians).
    '''
    # Set turbine geometry
    centerX = turbine.centerX
    centerY = turbine.centerY
    radius = turbine.r

    # Assemble global matrices
    Ax, Ay, theta = matrixAssemble(centerX, centerY, radius, ntheta)
    A = np.vstack([Ax, Ay])

    w0 = np.zeros(2 * len(theta))

    sol = root(residual, w0, args=(A, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager, z, H), tol=1e-6)
    if not sol.success:
        raise RuntimeError(f'Solver did not converge: {sol.message}')

    w = sol.x
    u, v = w[:len(theta)], w[len(theta):]

    q, ka, CT, CP, Rp, Tp, Zp, *_ = radialforce(u, v, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager, z, H)
    
    return CT, CP, Rp, Tp, Zp, theta

def initialize_turbine_and_environment(config):
    """
    Initializes turbine, environment, and solver objects from a configuration dictionary.

    Extracts geometric, operational, and fluid dynamics parameters from the provided
    configuration dictionary, instantiates the `Turbine` and `Environment` objects,
    and configures the aerodynamic evaluation model (e.g., file-based polars).

    Args:
        config (dict): Configuration dictionary containing required 'turbine',
            'environment', and 'solver' sections.

    Returns:
        tuple: A tuple containing:
            - turbine (Turbine): Instantiated and configured turbine object.
            - env (Environment): Instantiated fluid environment object.
            - simulation_params (dict): Parameters from the 'solver' section.
            - turbine_params (dict): Raw parameters from the 'turbine' section.
            - environment_params (dict): Raw parameters from the 'environment' section.
            - r (float): Turbine radius [m].
            - ntheta (int): Number of azimuthal discretization points.

    Raises:
        ValueError: If the solver method is set to 'file' but the file path
            ('solver.file.path') is not provided in the configuration.
    """
    turbine_params = config['turbine']
    environment_params = config['environment']
    simulation_params = config.get('solver', {})

    # 1. Enforce geometric consistency and scalar types for r, chord, and solidity
    r, chord, solidity = resolve_turbine_geometry(turbine_params, verbose=True)

    # 2. Extract turbine geometric & operational parameters (ensuring twist & delta are defined!)
    twist = turbine_params['twist']
    delta = turbine_params['delta']
    B = turbine_params['B']
    centerX = turbine_params['centerX']
    centerY = turbine_params['centerY']
    
    # Ensure Omega is a clean scalar float
    Omega = float(np.squeeze(turbine_params['Omega']))
    
    # 3. Extract environment parameters
    Vinf = np.array(environment_params['Vinf'])
    rho = environment_params['rho']
    mu = environment_params['mu']

    # 4. Instantiate objects safely
    turbine = Turbine(r, chord, twist, delta, B, Omega, centerX, centerY, solidity)
    env = Environment(Vinf, rho, mu)

    # 5. Configure Aerodynamics Solver
    method = simulation_params.get('method', 'neuralfoil')
    
    if method == 'file':
        file_cfg = simulation_params.get('file', {})
        filename = file_cfg.get('path') if isinstance(file_cfg, dict) else None
        
        if not filename:
            raise ValueError("solver.file.path is not defined in config, required when method='file'")
        
        turbine.aero = FileAerodynamics(filename)
    else:
        turbine.aero = None

    ntheta = turbine_params.get('ntheta', 360)

    return turbine, env, simulation_params, turbine_params, environment_params, r, ntheta

def run_simulation_case(params, base_config, flow_cfg=None, stall_angles=None, z=None, H=None):
    """
    Execute a single simulation case (2D mode or 3D blade section slice).

    Evaluates aerodynamic performance across a range of Tip Speed Ratios (TSR),
    computes performance curves (CP vs. TSR, CT vs. TSR), integrated force components,
    and optional azimuthal power coefficient distributions (Cp(theta)). Handles
    results serialization and plot generation.

    Parameters
    ----------
    params : tuple
        Case configuration tuple containing:
        - `airfoil_index` (int): Index referencing the airfoil profile.
        - `turbine_index` (int): Index referencing the turbine configuration.
        - `chord` (float): Blade chord length [m].
        - `solidity` (float): Rotor solidity [-].
        - `vinf` (float): Freestream wind velocity [m/s].
    base_config : dict
        Base simulation configuration dictionary containing solver, turbine,
        environment, and output settings.
    flow_cfg : dict, optional
        Configuration settings for the flow curvature submodel. If None, defaults
        to extracting settings from `base_config`.
    stall_angles : list of tuple of float, optional
        List of tuples containing positive and negative static stall angles of attack 
        `(aoa_stall_pos, aoa_stall_neg)` in radians for each airfoil.
    z : float, optional
        Vertical spanwise coordinate [m] for 3D slice evaluations. When provided,
        indicates 3D mode and bypasses 2D-only features (e.g., `cp_theta`).
    H : float, optional
        Total blade height/span [m] for 3D slice evaluations.

    Returns
    -------
    dict
        A dictionary containing simulation execution status and output metrics:
        - On success (`status='OK'`):
            - `name` (str): Output folder identifier string.
            - `status` (str): Execution status ('OK').
            - `time_sec` (float): Total execution duration in seconds.
            - `tsr` (np.ndarray): Array of evaluated Tip Speed Ratio values.
            - `CP` (np.ndarray): Power coefficient array.
            - `CT` (np.ndarray): Thrust coefficient array.
            - `Tp` (np.ndarray): Tangential force array [N/m].
            - `Rp` (np.ndarray): Radial force array [N/m].
            - `Zp` (np.ndarray): Spanwise force array [N/m].
        - On failure (`status='ERROR'`):
            - `name` (str): Output folder identifier string.
            - `status` (str): Execution status ('ERROR').
            - `error` (str): Cleaned error summary string.
            - `time_sec` (float): Execution duration prior to failure.
            - `traceback` (str): Truncated exception traceback string.

    Raises
    ------
    ValueError
        If `stall_angles` is None or if `fixed_parameter` in configuration is invalid.
    """
    # Start stopwatch
    start_time = time.perf_counter()

    airfoil_index, turbine_index, chord, solidity, vinf = params
    config = copy.deepcopy(base_config)

    # OUTPUT AND SAVING CONFIGURATIONS
    output_cfg = config.get('output', {})
    save_results = output_cfg.get('save', True)
    save_config_used = output_cfg.get('save_config', True)
    save_plot = output_cfg.get('save_plot', True)

    data_cfg = output_cfg.get('data_file', {})
    data_format = data_cfg.get('format', 'dat')
    include_header = data_cfg.get('include_header', True)

    plot_cfg = output_cfg.get('plot_image', {})
    image_format = plot_cfg.get('format', 'png')
    dpi = plot_cfg.get('dpi', 300)

    # SIMULATION MODE & CP(THETA) RULE VALIDATION
    is_3d_mode = config.get('simulation3d', {}).get('enabled', False) or (z is not None)
    cp_theta_cfg = output_cfg.get('cp_theta', {})
    cp_theta_requested = cp_theta_cfg.get('enabled', False)
    
    cp_theta_enabled = cp_theta_requested and not is_3d_mode
    if cp_theta_requested and is_3d_mode:
        print("--> [INFO] cp_theta extraction bypassed: Feature is only supported in 2D mode.", flush=True)

    target_tsr = cp_theta_cfg.get('target_tsr', 2.58)
    save_cp_theta_data = cp_theta_cfg.get('save_data', True)
    save_cp_theta_plot = cp_theta_cfg.get('save_plot', True)

    # TURBINE & ATMOSPHERIC PROPERTIES INITIALIZATION
    airfoil_name = config['solver']['neuralfoil']['airfoil'][airfoil_index]
    config['turbine']['chord'] = chord
    config['turbine']['solidity'] = solidity
    config['environment']['Vinf'] = vinf

    turbine, env, _, _, _, r, ntheta = initialize_turbine_and_environment(config)
    angular_velocity = config['turbine']['Omega']
    delta = config['turbine']['delta']

    # SUBMODEL SETUP (FLOW CURVATURE)
    if flow_cfg is None:
        flow_cfg = config.get('submodels', {}).get('flow_curvature', {})

    if flow_cfg.get('enabled', False):
        flow_manager = FlowCurvatureManager(
            chord=chord, 
            normalized_hook_point=flow_cfg.get('normalized_hook_point', 0.0),
            enabled=True
        )
    else:
        flow_manager = None

    def fmt(val):
        return str(val).replace('.', 'p')

    folder_name = (
        f'{airfoil_name}_ch{fmt(chord)}_sol{fmt(solidity)}_vinf{fmt(vinf)}'
        f'_delta{fmt(delta)}_r{fmt(r)}'
    )
    result_dir = os.path.join('src/results/temporary_results', folder_name)
    config['output']['result_folder'] = folder_name

    if save_results or save_config_used or save_plot:
        os.makedirs(result_dir, exist_ok=True)

    if save_config_used:
        from src.pyvawt.single.utils import save_config
        save_config(config, os.path.join(result_dir, 'config_used.yaml'))

    def _to_scalar(val):
        try:
            return val[0]
        except (TypeError, IndexError):
            return val

    fixed_parameter = config['solver']['fixed_parameter']
    ntheta = config['turbine']['ntheta']
    aero_method = config.get('solver', {}).get('method', 'neuralfoil')

    if aero_method == 'neuralfoil':
        turbine.aero = NeuralFoilAerodynamics(turbine_index=turbine_index, airfoil_index=airfoil_index, config=config)

    if stall_angles is None:
        raise ValueError('Stall angles must be provided')

    aoaStallPos, aoaStallNeg = stall_angles[airfoil_index]
    turbine.aero.aoaStallPos = aoaStallPos
    turbine.aero.aoaStallNeg = aoaStallNeg

    try:
        tsr_cfg = config.get('solver', {}).get('tsr', {})
        tsr_min = float(tsr_cfg.get('min', 1.0))
        tsr_max = float(tsr_cfg.get('max', 7.0))
        n = int(tsr_cfg.get('n_points', 20))
        tsrvec = np.linspace(tsr_min, tsr_max, n)
        
        CPvec, CTvec = np.zeros(n), np.zeros(n)
        Rpvec, Tpvec, Zpvec = np.zeros(n), np.zeros(n), np.zeros(n)

        if cp_theta_enabled:
            target_idx = int(np.argmin(np.abs(tsrvec - target_tsr)))
            actual_target_tsr = tsrvec[target_idx]
        else:
            target_idx = -1
            actual_target_tsr = None

        cp_theta_file = os.path.join(result_dir, "cp_theta_distribution.dat")
        should_write_cp_theta_file = save_results and cp_theta_enabled and save_cp_theta_data

        if should_write_cp_theta_file:
            with open(cp_theta_file, "w") as f:
                f.write("TSR\ttheta_deg\tCp_theta\n")

        has_plot_data = False
        theta_plot, cp_plot = None, None

        for i, tsr in enumerate(tsrvec):
            if fixed_parameter == 'vinf':
                turbine.Omega = vinf * tsr / _to_scalar(r)
            elif fixed_parameter == 'omega':
                turbine.Omega = _to_scalar(angular_velocity)
                env.Vinf = turbine.Omega * _to_scalar(r) / tsr
            else:
                raise ValueError("Invalid value for 'fixed_parameter'. Use 'vinf' or 'omega'.")

            CT, CP, Rp, Tp_raw, Zp, theta = actuatorcylinder(
                turbine, env, ntheta, config, turbine_index, airfoil_index, flow_manager, z, H
            )

            CPvec[i], CTvec[i] = CP, CT
            Rpvec[i], Tpvec[i], Zpvec[i] = _to_scalar(Rp), _to_scalar(Tp_raw), _to_scalar(Zp)

            if cp_theta_enabled and i == target_idx:
                theta_deg = np.degrees(theta)
                Href = 1.0
                Sref = 2 * _to_scalar(turbine.r) * Href

                Cp_theta = (abs(turbine.Omega) * _to_scalar(turbine.r) * Tp_raw) / (
                    0.5 * env.rho * _to_scalar(env.Vinf)**3 * Sref
                )

                theta_plot = theta_deg.copy()
                cp_plot = Cp_theta.copy()
                has_plot_data = True

                if should_write_cp_theta_file:
                    Cp_theta_iterable = np.full_like(theta_deg, Cp_theta) if not hasattr(Cp_theta, "__iter__") else Cp_theta
                    with open(cp_theta_file, "a") as f:
                        for th, cp in zip(theta_deg, Cp_theta_iterable):
                            f.write(f"{tsr:.6f}\t{th:.6f}\t{cp:.6f}\n")

        data_to_save = np.column_stack((tsrvec, CPvec, CTvec, Rpvec, Tpvec, Zpvec))
        header = 'TSR\tCP\tCT\tRp\tTp\tZp'
        if save_results:
            filename = f'results_{airfoil_name}.{data_format}'
            filepath = os.path.join(result_dir, filename)

            if data_format == 'dat':
                np.savetxt(filepath, data_to_save, header=header if include_header else '', fmt='%.6f', delimiter='\t')
            elif data_format == 'csv':
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if include_header:
                        writer.writerow(header.split('\t'))
                    writer.writerows(data_to_save.tolist())

        if save_plot:
            fig1 = plt.figure()
            plt.plot(tsrvec, CPvec, 'o-')
            plt.xlabel('TSR')
            plt.ylabel('$C_p$')
            plt.grid(True)
            plt.tight_layout()

            fig1_filename = f'cp_curve_{airfoil_name}.{image_format}'
            fig1.savefig(os.path.join(result_dir, fig1_filename), format=image_format, dpi=dpi)
            plt.close(fig1)

        if save_plot and cp_theta_enabled and save_cp_theta_plot and has_plot_data:
            fig2 = plt.figure()
            plt.plot(theta_plot, cp_plot, 'o-')
            plt.xlabel('Azimuthal angle (deg)')
            plt.ylabel('$C_p(\\theta)$')
            plt.title(f'TSR = {actual_target_tsr:.2f}')
            plt.grid(True)
            plt.tight_layout()

            fig2_filename = f'cp_theta_{airfoil_name}_tsr{fmt(round(actual_target_tsr, 2))}.{image_format}'
            fig2.savefig(os.path.join(result_dir, fig2_filename), format=image_format, dpi=dpi)
            plt.close(fig2)

        elapsed = time.perf_counter() - start_time
        return {
            'name': folder_name, 'status': 'OK', 'time_sec': round(elapsed, 2),
            'tsr': tsrvec, 'CP': CPvec, 'CT': CTvec, 'Tp': Tpvec, 'Rp': Rpvec, 'Zp': Zpvec
        }

    except Exception as e:
            elapsed = time.perf_counter() - start_time
            # Clean up multi-line error messages into a single concise string
            clean_error = str(e).strip().split('\n')[0]

            return {
                'name': folder_name,
                'status': 'ERROR',
                'error': clean_error,
                'time_sec': round(elapsed, 2),
                'traceback': traceback.format_exc(limit=2),
            }

def simulate_3D_turbine(
    base_config: dict[str, Any],
    stall_angles: list[tuple[float, float]] | dict[int, tuple[float, float]],
) -> dict[str, Any]:
    """
    Execute a multi-slice 3D Vertical Axis Wind Turbine (VAWT) simulation.

    Discretizes the turbine rotor vertically into horizontal 2D slices, evaluates
    aerodynamic performance per slice under vertical wind shear profiles (e.g.,
    power-law velocity profile), integrates total power output across all slices,
    and computes the global 3D power coefficient (Cp).

    Parameters
    ----------
    base_config : dict
        Base simulation configuration dictionary containing turbine parameters,
        environment properties, solver settings, and 3D discretization options.
    stall_angles : list of tuple of float or dict
        Collection of positive and negative static stall angles of attack [rad]
        indexed by airfoil profile.

    Returns
    -------
    dict
        Dictionary containing integrated 3D simulation results:
        - `tsr` (np.ndarray or None): Global Tip Speed Ratio vector.
        - `cp_3d` (np.ndarray or None): Global 3D power coefficient array.
        - `result_dir` (str): Path to the 3D results output directory.
        - `elapsed_time` (float): Total execution duration in seconds.
        - `failed_slices` (list of tuple): List of `(slice_index, error_summary)`
          tuples for any slices that encountered solver errors.
        If 3D mode is disabled in `base_config`, returns the single 2D dictionary
        output from `run_simulation_case`.
    """
    start_time_3d = time.perf_counter()
    UI.section("3D SIMULATION EXECUTION")

    # Extract 3D configurations
    sim3d_cfg = base_config.get("solver", {}).get("simulation3d", {})
    sim3d_settings = sim3d_cfg.get("settings", {})

    # Fallback to 2D standard mode if 3D is disabled
    if not sim3d_cfg.get("enabled", False):
        UI.status(
            "Mode Notice",
            "3D mode disabled. Running single 2D simulation...",
            level="warn",
        )
        airfoil_index = 0
        turbine_index = 0
        chord = base_config["turbine"]["chord"][0]
        solidity = base_config["turbine"]["solidity"][0]
        vinf = base_config["environment"]["Vinf"][0]
        return run_simulation_case(
            params=(airfoil_index, turbine_index, chord, solidity, vinf),
            base_config=base_config,
            stall_angles=stall_angles,
        )

    # Disable file I/O inside individual slice loops for maximum speed
    config_no_output = copy.deepcopy(base_config)
    config_no_output["output"]["save"] = False
    config_no_output["output"]["save_plot"] = False
    config_no_output["output"]["save_config"] = False

    # Retrieve parameters
    height = float(base_config.get("turbine", {}).get("height", 20.0))
    n_slices = int(sim3d_settings.get("vertical_layers", 20))
    velocity_profile = sim3d_settings.get("velocity_profile", "power_law")

    airfoil_index = sim3d_settings.get("airfoil_index", 0)
    turbine_index = sim3d_settings.get("turbine_index", 0)
    chord = base_config["turbine"]["chord"][0]
    solidity = base_config["turbine"]["solidity"][0]

    # Setup 3D output directory
    folder_name_3D = f"3D_H{height}_Ns{n_slices}"
    result_dir_3D = os.path.join("src/results/results_3D", folder_name_3D)
    os.makedirs(result_dir_3D, exist_ok=True)

    # Display execution parameters
    UI.status("Turbine Height (H)", f"{height:.2f} m")
    UI.status("Vertical Layers (Slices)", f"{n_slices}")
    UI.status("Wind Profile", f"{velocity_profile.upper()}")
    print()

    # Environmental and geometrical properties
    rho = base_config["environment"]["rho"]
    r = base_config["turbine"]["r"]
    Vr = base_config["environment"]["Vinf"][0]
    Zr = height / 2.0
    alpha = 0.13

    # Power-law grid discretization
    beta = sim3d_settings.get("discretization_power", 2.0)
    eta = np.linspace(0, 1, n_slices + 1)
    z_nodes = height * (1.0 - (1.0 - eta) ** beta)
    z_centers = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    dz_array = z_nodes[1:] - z_nodes[:-1]

    power_total = None
    tsrvec_global = None
    failed_slices = []
    start_slices_loop = time.perf_counter()

    # Execute vertical slice loop
    for i in range(n_slices):
        z = z_centers[i]
        dz = dz_array[i]

        vinf = Vr * (z / Zr) ** alpha if velocity_profile == "power_law" else Vr
        z_centered = z - (height / 2.0)

        result = run_simulation_case(
            params=(airfoil_index, turbine_index, chord, solidity, vinf),
            base_config=config_no_output,
            stall_angles=stall_angles,
            z=z_centered,
            H=height,
        )

        if result["status"] == "OK":
            CP = np.array(result["CP"])
            A_slice = 2.0 * r * dz
            power_slice = CP * (0.5 * rho * (vinf**3) * A_slice)

            if power_total is None:
                power_total = power_slice.copy()
                tsrvec_global = np.array(result["tsr"])
            else:
                power_total += power_slice
        else:
            # Format and collect error string concisely
            raw_err = result.get("error", "Unknown solver error")
            short_err = raw_err.replace("RuntimeError: ", "").strip()
            failed_slices.append((i + 1, short_err))

        # Dynamic progress bar updates smoothly without getting broken by error prints
        elapsed_loop = time.perf_counter() - start_slices_loop
        UI.progress_bar(i + 1, n_slices, elapsed_loop, prefix="3D Progress")

    # Compute global Cp
    total_time_3d = time.perf_counter() - start_time_3d
    A_total = 2.0 * r * height
    P_available = 0.5 * rho * (Vr**3) * A_total
    Cp_3D = (power_total / P_available) if power_total is not None else None

    # Display summary card with error breakdown
    UI.section("3D SIMULATION RESULTS")

    if not failed_slices:
        UI.status("Status", "Completed Successfully", level="ok")
    else:
        status_msg = f"Completed with Warnings ({len(failed_slices)}/{n_slices} Slices Failed)"
        UI.status("Status", status_msg, level="warn")

    # High-precision execution time display
    UI.status("Total Execution Time", UI.format_time(total_time_3d))
    UI.status("Output Directory", result_dir_3D)

    # Detailed breakdown for failed slices
    if failed_slices:
        print(f"\n  {UI.YELLOW}{UI.BOLD}Failed Slices Breakdown:{UI.RESET}")
        for slice_num, err in failed_slices:
            # Truncate long error messages to 75 characters for alignment
            truncated_err = (err[:75] + "...") if len(err) > 75 else err
            print(f"    {UI.RED}• Slice {slice_num:02d}{UI.RESET} : {truncated_err}")
        print()

    return {
        "tsr": tsrvec_global,
        "cp_3d": Cp_3D,
        "result_dir": result_dir_3D,
        "elapsed_time": round(total_time_3d, 2),
        "failed_slices": failed_slices,
    }
# ======== Auxiliary Methods =========

def trapz(x, y):
    '''
    Computes the integral of a function using the trapezoidal rule.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x-values (independent variable).
    y : numpy.ndarray
        Array of y-values (dependent variable), corresponding to the function values at the x-values.

    Returns
    -------
    float
        The computed integral of the function using the trapezoidal rule.
    '''
    integral = 0.0
    for i in range(len(x) - 1):
        integral += (x[i+1] - x[i]) * 0.5 * (y[i] + y[i+1])
    return integral

def pInt(theta, f):
    '''
    Computes the integral of a periodic function using the trapezoidal rule, considering periodic boundary conditions.

    Parameters
    ----------
    theta : numpy.ndarray
        Array of angular values representing the independent variable.
    f : numpy.ndarray
        Array of function values corresponding to the function evaluated at the points in `theta`.

    Returns
    -------
    float
        The computed integral, including the periodic boundary contribution.
    '''
    # Compute the integral using the trapezoidal rule
    integral = trapz(theta, f)

    # Add the contribution from the periodic boundary points
    dtheta = 2 * theta[0] # Assume equal spacing, starts at 0
    integral += dtheta * 0.5 * (f[0] + f[-1])

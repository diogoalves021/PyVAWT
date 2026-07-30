"""
Actuator Cylinder Aerodynamic Solver for Coupled VAWT Turbines.

Implements Linearized and Coupled Actuator Cylinder theory for Vertical Axis
Wind Turbines (VAWTs), featuring Gauss-Legendre panel integrations, HDF5 matrix
precomputations, Numba-compiled kernels, and SciPy non-linear system solving.
"""
from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Callable, NamedTuple

import h5py
import numpy as np
from numba import njit
from scipy.optimize import root

# Logging configuration
logger = logging.getLogger(__name__)

# ==============================================================================
# GLOBAL RAM CACHE & GAUSS-LEGENDRE QUADRATURE NODES
# ==============================================================================

_MATRIX_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

GL_NODES: np.ndarray = np.array([
    -0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
    -0.4333953941292472, -0.1488743389816312,  0.1488743389816312,
     0.4333953941292472,  0.6794095682990244,  0.8650633666889845,
     0.9739065285171717
], dtype=np.float64)

GL_WEIGHTS: np.ndarray = np.array([
    0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
    0.2692667193099963, 0.2955242247147529, 0.2955242247147529,
    0.2692667193099963, 0.2190863625159820, 0.1494513491505806,
    0.0666713443086881
], dtype=np.float64)


# ==============================================================================
# DOMAIN DATA STRUCTURES
# ==============================================================================

class Turbine:
    """
    Physical and operational representation of a Vertical Axis Wind Turbine (VAWT).

    Parameters
    ----------
    r : float
        Rotor radius in meters ($m$).
    chord : float
        Blade chord length in meters ($m$).
    twist : float
        Blade pitch / twist angle in radians.
    delta : float
        Catenary or coning inclination angle in radians.
    B : int
        Number of rotor blades.
    af : Callable[[np.ndarray | float], tuple[np.ndarray | float, np.ndarray | float]]
        Airfoil polar evaluation callable returning `(cl, cd)` for query angle `alpha`.
    Omega : float
        Rotor angular rotational speed in $rad/s$.
    centerX : float
        Rotor hub center position along the X-axis ($m$).
    centerY : float
        Rotor hub center position along the Y-axis ($m$).

    Attributes
    ----------
    r : float
        Rotor radius ($m$).
    chord : float
        Blade chord length ($m$).
    twist : float
        Blade pitch angle (rad).
    delta : float
        Coning angle (rad).
    B : int
        Blade count.
    af : Callable
        Airfoil lookup callback function.
    Omega : float
        Rotational speed ($rad/s$).
    centerX : float
        Hub X coordinate ($m$).
    centerY : float
        Hub Y coordinate ($m$).
    """

    def __init__(
        self,
        r: float,
        chord: float,
        twist: float,
        delta: float,
        B: int,
        af: Callable[[np.ndarray | float], tuple[np.ndarray | float, np.ndarray | float]],
        Omega: float,
        centerX: float,
        centerY: float
    ) -> None:
        self.r = float(r)
        self.chord = float(chord)
        self.twist = float(twist)
        self.delta = float(delta)
        self.B = int(B)
        self.af = af
        self.Omega = float(Omega)
        self.centerX = float(centerX)
        self.centerY = float(centerY)


class Environment:
    """
    Fluid domain physical properties container.

    Parameters
    ----------
    Vinf : float
        Free stream inflow velocity in $m/s$.
    rho : float
        Fluid mass density in $kg/m^3$.
    mu : float
        Dynamic fluid viscosity in $Pa \cdot s$.

    Attributes
    ----------
    Vinf : float
        Free stream velocity ($m/s$).
    rho : float
        Fluid density ($kg/m^3$).
    mu : float
        Dynamic viscosity ($Pa \cdot s$).
    """

    def __init__(self, Vinf: float, rho: float, mu: float) -> None:
        self.Vinf = float(Vinf)
        self.rho = float(rho)
        self.mu = float(mu)


# ==============================================================================
# NUMBA ACCELERATED NUMERICAL INTEGRATION KERNELS
# ==============================================================================

@njit(fastmath=True)
def fast_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Accelerated 1D trapezoidal numerical integration via Numba.

    Parameters
    ----------
    y : np.ndarray
        1D array of integrand function values.
    x : np.ndarray
        1D array of monotonic sample coordinates corresponding to `y`.

    Returns
    -------
    float
        Approximated definite integral value.
    """
    n = len(x)
    total = 0.0
    for i in range(n - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


@njit(fastmath=True)
def Dxintegrand(x: float, y: float, phi: float) -> float:
    """
    Computes kernel integrand for axial velocity influence ($D_x$).

    Parameters
    ----------
    x : float
        Normalized target evaluation point X coordinate.
    y : float
        Normalized target evaluation point Y coordinate.
    phi : float
        Source panel azimuthal location angle in radians.

    Returns
    -------
    float
        Kernel evaluation scalar value.
    """
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)
    denom = 2.0 * math.pi * (v1 * v1 + v2 * v2)
    if denom == 0.0:
        return 0.0
    return (v1 * math.sin(phi) - v2 * math.cos(phi)) / denom


@njit(fastmath=True)
def Ayintegrand(x: float, y: float, phi: float) -> float:
    """
    Computes kernel integrand for lateral velocity influence ($A_y$).

    Parameters
    ----------
    x : float
        Normalized target evaluation point X coordinate.
    y : float
        Normalized target evaluation point Y coordinate.
    phi : float
        Source panel azimuthal location angle in radians.

    Returns
    -------
    float
        Kernel evaluation scalar value.
    """
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)
    if abs(v1) < 1e-12 and abs(v2) < 1e-12:
        return 0.0
    denom = 2.0 * math.pi * (v1 * v1 + v2 * v2)
    return (v1 * math.cos(phi) + v2 * math.sin(phi)) / denom


@njit(fastmath=True)
def panelIntegration_numba(
    xvec: np.ndarray,
    yvec: np.ndarray,
    thetavec: np.ndarray,
    is_ay: bool
) -> np.ndarray:
    """
    Performs 10-point Gauss-Legendre panel integration over rotor boundary.

    Parameters
    ----------
    xvec : np.ndarray
        Array of target evaluation point X coordinates.
    yvec : np.ndarray
        Array of target evaluation point Y coordinates.
    thetavec : np.ndarray
        Azimuthal grid angles array.
    is_ay : bool
        If True, evaluates lateral kernel $A_y$; otherwise evaluates axial kernel $D_x$.

    Returns
    -------
    np.ndarray
        Influence matrix of shape `(len(xvec), len(thetavec))`.
    """
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0]
    half_dtheta = dtheta / 2.0
    A = np.zeros((nx, ntheta), dtype=np.float64)

    for i in range(nx):
        xi = xvec[i]
        yi = yvec[i]
        for j in range(ntheta):
            m = thetavec[j]
            res = 0.0
            for k in range(10):
                phi = m + half_dtheta * GL_NODES[k]
                if is_ay:
                    val = Ayintegrand(xi, yi, phi)
                else:
                    val = Dxintegrand(xi, yi, phi)
                res += GL_WEIGHTS[k] * val
            A[i, j] = half_dtheta * res
    return A


def AyIJ(xvec: np.ndarray, yvec: np.ndarray, thetavec: np.ndarray) -> np.ndarray:
    """
    Computes lateral velocity influence matrix $A_{y,IJ}$ between turbine panels.

    Parameters
    ----------
    xvec : np.ndarray
        Target point X coordinates.
    yvec : np.ndarray
        Target point Y coordinates.
    thetavec : np.ndarray
        Source panel azimuthal angles.

    Returns
    -------
    np.ndarray
        Transverse influence matrix.
    """
    return panelIntegration_numba(xvec, yvec, thetavec, True)


def DxIJ(xvec: np.ndarray, yvec: np.ndarray, thetavec: np.ndarray) -> np.ndarray:
    """
    Computes axial velocity dipole influence matrix $D_{x,IJ}$ between turbine panels.

    Parameters
    ----------
    xvec : np.ndarray
        Target point X coordinates.
    yvec : np.ndarray
        Target point Y coordinates.
    thetavec : np.ndarray
        Source panel azimuthal angles.

    Returns
    -------
    np.ndarray
        Axial dipole influence matrix.
    """
    return panelIntegration_numba(xvec, yvec, thetavec, False)


@njit(fastmath=True)
def WxIJ(xvec: np.ndarray, yvec: np.ndarray, thetavec: np.ndarray) -> np.ndarray:
    """
    Computes wake downwash velocity influence matrix $W_{x,IJ}$.

    Parameters
    ----------
    xvec : np.ndarray
        Target evaluation X coordinates.
    yvec : np.ndarray
        Target evaluation Y coordinates.
    thetavec : np.ndarray
        Azimuthal grid angles.

    Returns
    -------
    np.ndarray
        Wake influence matrix of shape `(len(xvec), len(thetavec))`.
    """
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0]
    Wx = np.zeros((nx, ntheta), dtype=np.float64)
    theta_bounds = thetavec + dtheta / 2.0

    for i in range(nx):
        x = xvec[i]
        y = yvec[i]
        if -1.0 <= y <= 1.0 and x >= 0.0 and (x * x + y * y) >= 1.0:
            thetak = math.acos(y)
            k = np.searchsorted(theta_bounds, thetak, side='right')
            if 0 <= k < ntheta:
                Wx[i, k] = -1.0
                Wx[i, ntheta - k - 1] = 1.0
    return Wx


@njit(fastmath=True)
def DxII(thetavec: np.ndarray) -> np.ndarray:
    """
    Computes self-influence axial dipole matrix $D_{x,II}$ for a single rotor.

    Parameters
    ----------
    thetavec : np.ndarray
        Rotor azimuthal discretization vector.

    Returns
    -------
    np.ndarray
        Self-influence matrix of shape `(ntheta, ntheta)`.
    """
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0]
    Rx = (dtheta / (4.0 * np.pi)) * np.ones((ntheta, ntheta), dtype=np.float64)
    half_n = ntheta // 2
    inv_n = 1.0 / ntheta
    for i in range(ntheta):
        if i < half_n:
            Rx[i, i] = (-1.0 + inv_n) / 2.0
        else:
            Rx[i, i] = (1.0 + inv_n) / 2.0
    return Rx


@njit(fastmath=True)
def WxII(thetavec: np.ndarray) -> np.ndarray:
    """
    Computes self-influence wake downwash matrix $W_{x,II}$ for a single rotor.

    Parameters
    ----------
    thetavec : np.ndarray
        Rotor azimuthal discretization vector.

    Returns
    -------
    np.ndarray
        Self-influence wake matrix of shape `(ntheta, ntheta)`.
    """
    ntheta = len(thetavec)
    Wx = np.zeros((ntheta, ntheta), dtype=np.float64)
    half_n = ntheta // 2
    for i in range(half_n, ntheta):
        Wx[i, ntheta - 1 - i] = -1.0
    return Wx


# ==============================================================================
# MATRIX PRECOMPUTATION & ASSEMBLY
# ==============================================================================

def precomputeMatrices(ntheta: int, modulepath: str | Path) -> Path:
    """
    Precomputes single-turbine self-influence matrices and compresses to HDF5.

    Parameters
    ----------
    ntheta : int
        Number of azimuthal discretizations.
    modulepath : str or Path
        Target output directory path for storage.

    Returns
    -------
    Path
        Resolved file path to the created `.h5` file.
    """
    dtheta = 2.0 * np.pi / ntheta
    theta = np.arange(dtheta / 2.0, 2.0 * np.pi, dtheta)

    Dxself = DxII(theta)
    Wxself = WxII(theta)
    Ayself = AyIJ(-np.sin(theta), np.cos(theta), theta)

    filepath = Path(modulepath) / f"theta-{ntheta}.h5"
    with h5py.File(filepath, "w") as file:
        file.create_dataset("theta", data=theta)
        file.create_dataset("Dx", data=Dxself)
        file.create_dataset("Wx", data=Wxself)
        file.create_dataset("Ay", data=Ayself)

    return filepath


def matrixAssemble(
    centerX: np.ndarray | list[float],
    centerY: np.ndarray | list[float],
    radii: np.ndarray | list[float],
    ntheta: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assembles coupled influence coefficient matrices (Ax, Ay) for turbine array.

    Utilizes an in-memory RAM cache to avoid redundant HDF5 file read operations.

    Parameters
    ----------
    centerX : array_like
        X-coordinates of turbine hub centers ($m$).
    centerY : array_like
        Y-coordinates of turbine hub centers ($m$).
    radii : array_like
        Turbine rotor radii ($m$).
    ntheta : int
        Azimuthal grid points count per rotor.

    Returns
    -------
    Ax : np.ndarray
        Combined axial influence matrix `Dx + Wx` of shape `(N, N)`.
    Ay : np.ndarray
        Transverse influence matrix of shape `(N, N)`.
    theta : np.ndarray
        Azimuthal discretization grid of shape `(ntheta,)`.
    """
    cache_key = (tuple(centerX), tuple(centerY), tuple(radii), ntheta)
    if cache_key in _MATRIX_CACHE:
        return _MATRIX_CACHE[cache_key]

    file_name = f"theta-{ntheta}.h5"
    modulepath = Path.cwd()
    filepath = modulepath / file_name

    if not filepath.is_file():
        filepath = precomputeMatrices(ntheta, modulepath)

    with h5py.File(filepath, "r") as f:
        theta = f["theta"][:]
        Dxself = f["Dx"][:]
        Wxself = f["Wx"][:]
        Ayself = f["Ay"][:]

    nturbines = len(radii)
    Dx = np.zeros((nturbines * ntheta, nturbines * ntheta))
    Wx = np.zeros((nturbines * ntheta, nturbines * ntheta))
    Ay = np.zeros((nturbines * ntheta, nturbines * ntheta))

    for I in range(nturbines):
        for J in range(nturbines):
            x = (centerX[I] - radii[I] * np.sin(theta) - centerX[J]) / radii[J]
            y = (centerY[I] + radii[I] * np.cos(theta) - centerY[J]) / radii[J]

            if I == J:
                Dxsub = Dxself
                Wxsub = Wxself
                Aysub = Ayself
            elif J < I and radii[I] == radii[J]:
                Dxsub = Dx[J * ntheta:(J + 1) * ntheta, I * ntheta:(I + 1) * ntheta]
                Aysub = Ay[J * ntheta:(J + 1) * ntheta, I * ntheta:(I + 1) * ntheta]
                Wxsub = WxIJ(x, y, theta)
            else:
                Dxsub = DxIJ(x, y, theta)
                Wxsub = WxIJ(x, y, theta)
                Aysub = AyIJ(x, y, theta)

            Dx[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Dxsub
            Wx[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Wxsub
            Ay[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Aysub

    Ax = Dx + Wx
    result = (Ax, Ay, theta)
    _MATRIX_CACHE[cache_key] = result
    return result


# ==============================================================================
# AERODYNAMIC KINEMATICS & SOLVER KERNELS
# ==============================================================================

@njit(fastmath=True)
def _radialforce_kinematics(
    uvec: np.ndarray,
    vvec: np.ndarray,
    thetavec: np.ndarray,
    Vinf: float,
    Omega: float,
    r: float,
    twist: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates local apparent velocities, inflow angles, and angles of attack.

    Parameters
    ----------
    uvec : np.ndarray
        Axial induction velocity distribution vector.
    vvec : np.ndarray
        Transverse induction velocity distribution vector.
    thetavec : np.ndarray
        Azimuthal grid points array.
    Vinf : float
        Free stream velocity ($m/s$).
    Omega : float
        Rotor angular velocity ($rad/s$).
    r : float
        Rotor radius ($m$).
    twist : float
        Blade twist angle (rad).

    Returns
    -------
    rotation : float
        Direction multiplier (+1.0 for counter-clockwise, -1.0 for clockwise).
    W : np.ndarray
        Local apparent velocity magnitude vector ($m/s$).
    phi : np.ndarray
        Local flow angle vector (rad).
    alpha : np.ndarray
        Local effective angle of attack vector (rad).
    """
    rotation = 1.0 if Omega >= 0.0 else -1.0
    abs_Omega = abs(Omega)
    n = len(thetavec)

    W = np.empty(n, dtype=np.float64)
    phi = np.empty(n, dtype=np.float64)
    alpha = np.empty(n, dtype=np.float64)

    for i in range(n):
        sin_t = math.sin(thetavec[i])
        cos_t = math.cos(thetavec[i])
        u_i = uvec[i]
        v_i = vvec[i]

        vn = Vinf * (1.0 + u_i) * sin_t - Vinf * v_i * cos_t
        vt = rotation * (Vinf * (1.0 + u_i) * cos_t + Vinf * v_i * sin_t) + abs_Omega * r
        w = math.sqrt(vn * vn + vt * vt)
        p = math.atan2(vn, vt)

        W[i] = w
        phi[i] = p
        alpha[i] = p - twist

    return rotation, W, phi, alpha


@njit(fastmath=True)
def _radialforce_postprocess(
    cl: np.ndarray,
    cd: np.ndarray,
    phi: np.ndarray,
    W: np.ndarray,
    thetavec: np.ndarray,
    Vinf: float,
    rho: float,
    r: float,
    chord: float,
    B: int,
    rotation: float,
    delta: float,
    Omega: float
) -> tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes non-dimensional forces, Glauert correction factors, and global C_T/C_P.

    Parameters
    ----------
    cl : np.ndarray
        Lift coefficients vector.
    cd : np.ndarray
        Drag coefficients vector.
    phi : np.ndarray
        Local flow angles vector.
    W : np.ndarray
        Local apparent velocities vector.
    thetavec : np.ndarray
        Azimuthal grid points vector.
    Vinf : float
        Free stream inflow velocity ($m/s$).
    rho : float
        Fluid density ($kg/m^3$).
    r : float
        Rotor radius ($m$).
    chord : float
        Blade chord length ($m$).
    B : int
        Blade count.
    rotation : float
        Rotor rotation direction (+1.0 / -1.0).
    delta : float
        Coning angle (rad).
    Omega : float
        Rotational speed ($rad/s$).

    Returns
    -------
    q : np.ndarray
        Induction source distribution vector.
    ka : float
        Glauert induction correction multiplier.
    CT : float
        Rotor Thrust Coefficient.
    CP : float
        Rotor Power Coefficient.
    Rp : np.ndarray
        Radial force distribution ($N/m$).
    Tp : np.ndarray
        Tangential force distribution ($N/m$).
    Zp : np.ndarray
        Vertical/Axial force distribution ($N/m$).
    """
    n = len(thetavec)
    q = np.empty(n, dtype=np.float64)
    Rp = np.empty(n, dtype=np.float64)
    Tp = np.empty(n, dtype=np.float64)
    Zp = np.empty(n, dtype=np.float64)
    integrand = np.empty(n, dtype=np.float64)

    sigma = B * chord / r
    cos_delta = math.cos(delta)
    tan_delta = math.tan(delta)
    abs_Omega = abs(Omega)
    inv_vinf = 1.0 / Vinf

    for i in range(n):
        p = phi[i]
        w = W[i]
        cos_p = math.cos(p)
        sin_p = math.sin(p)

        cn_i = cl[i] * cos_p + cd[i] * sin_p
        ct_i = cl[i] * sin_p - cd[i] * cos_p

        w_ratio = w * inv_vinf
        w_ratio_sq = w_ratio * w_ratio

        q[i] = (sigma / (4.0 * math.pi)) * cn_i * w_ratio_sq

        qd = 0.5 * rho * w * w
        Rp[i] = -cn_i * qd * chord
        Tp[i] = ct_i * qd * chord / cos_delta
        Zp[i] = -cn_i * qd * chord * tan_delta

        sin_t = math.sin(thetavec[i])
        cos_t = math.cos(thetavec[i])
        integrand[i] = w_ratio_sq * (cn_i * sin_t - rotation * ct_i * cos_t / cos_delta)

    CT = (sigma / (4.0 * math.pi)) * fast_trapz(integrand, thetavec)

    # Correction factor
    if CT > 2.0:
        a = 0.5 * (1.0 + math.sqrt(1.0 + CT))
        ka = 1.0 / (a - 1.0)
    elif CT > 0.96:
        a = (1.0 / 7.0) * (1.0 + 3.0 * math.sqrt(3.5 * CT - 3.0))
        ka = 18.0 * a / (7.0 * a * a - 2.0 * a + 4.0)
    else:
        a = 0.5 * (1.0 - math.sqrt(1.0 - CT))
        ka = 1.0 / (1.0 - a)

    H = 1.0
    Sref = 2.0 * r * H
    Q = r * Tp
    P = abs_Omega * B / (2.0 * math.pi) * fast_trapz(Q, thetavec)
    CP = P / (0.5 * rho * Vinf**3 * Sref)

    return q, ka, CT, CP, Rp, Tp, Zp


def radialforce(
    uvec: np.ndarray,
    vvec: np.ndarray,
    thetavec: np.ndarray,
    turbine: Turbine,
    env: Environment
) -> tuple[np.ndarray, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-level orchestration wrapper for turbine radial force calculation.

    Parameters
    ----------
    uvec : np.ndarray
        Axial induction velocities vector.
    vvec : np.ndarray
        Transverse induction velocities vector.
    thetavec : np.ndarray
        Azimuthal grid points vector.
    turbine : Turbine
        Target physical turbine object.
    env : Environment
        Fluid domain environment properties.

    Returns
    -------
    tuple
        Tuple containing `(q, ka, CT, CP, Rp, Tp, Zp)`.
    """
    rotation, W, phi, alpha = _radialforce_kinematics(
        uvec, vvec, thetavec, env.Vinf, turbine.Omega, turbine.r, turbine.twist
    )
    cl, cd = turbine.af(alpha)
    return _radialforce_postprocess(
        cl, cd, phi, W, thetavec, env.Vinf, env.rho,
        turbine.r, turbine.chord, turbine.B, rotation, turbine.delta, turbine.Omega
    )


@njit(fastmath=True)
def _compute_residual_fast_single(
    A: np.ndarray,
    q: np.ndarray,
    ka: float,
    w: np.ndarray
) -> np.ndarray:
    """
    High-performance zero-allocation residual solver kernel for a single turbine.

    Parameters
    ----------
    A : np.ndarray
        Combined block influence matrix `[Ax; Ay]`.
    q : np.ndarray
        Source strength vector.
    ka : float
        Glauert induction correction factor scalar.
    w : np.ndarray
        Query induction state vector `[u, v]`.

    Returns
    -------
    np.ndarray
        Residual evaluation array.
    """
    res = A @ q
    for i in range(len(w)):
        res[i] = res[i] * ka - w[i]
    return res


@njit(fastmath=True)
def _compute_residual_output(
    A: np.ndarray,
    q: np.ndarray,
    k: np.ndarray,
    ntheta: int,
    w: np.ndarray
) -> np.ndarray:
    """
    Evaluates multi-turbine coupled residual vector.

    Parameters
    ----------
    A : np.ndarray
        Full multi-turbine influence matrix block `[Ax; Ay]`.
    q : np.ndarray
        Coupled source strength vector.
    k : np.ndarray
        Glauert correction factors array per turbine.
    ntheta : int
        Azimuthal points count.
    w : np.ndarray
        Combined induction query vector.

    Returns
    -------
    np.ndarray
        Coupled residual vector.
    """
    nturbines = len(k)
    kmult = np.empty(2 * nturbines * ntheta, dtype=np.float64)
    for i in range(nturbines):
        ki = k[i]
        start_u = i * ntheta
        end_u = (i + 1) * ntheta
        start_v = nturbines * ntheta + i * ntheta
        end_v = nturbines * ntheta + (i + 1) * ntheta

        kmult[start_u:end_u] = ki
        kmult[start_v:end_v] = ki

    return (A @ q) * kmult - w


def residual(
    w: np.ndarray,
    A: np.ndarray,
    theta: np.ndarray,
    k: np.ndarray,
    turbines: list[Turbine],
    env: Environment
) -> np.ndarray:
    """
    SciPy root-finder wrapper for evaluating coupled multi-turbine system residuals.

    Parameters
    ----------
    w : np.ndarray
        Current state vector containing `[u1, u2..., v1, v2...]`.
    A : np.ndarray
        Stacked full system influence matrix `[Ax; Ay]`.
    theta : np.ndarray
        Azimuthal discretizations array.
    k : np.ndarray
        Array of Glauert factors for each turbine.
    turbines : list[Turbine]
        List of target physical turbine objects.
    env : Environment
        Fluid domain properties.

    Returns
    -------
    np.ndarray
        Residual values array matching `w` shape.
    """
    ntheta = len(theta)
    nturbines = int(len(w) / (2 * ntheta))
    q = np.empty(ntheta * nturbines, dtype=np.float64)

    for i in range(nturbines):
        u = w[i * ntheta : (i + 1) * ntheta]
        v = w[nturbines * ntheta + i * ntheta : nturbines * ntheta + (i + 1) * ntheta]
        q_i, ka, *_ = radialforce(u, v, theta, turbines[i], env)
        q[i * ntheta : (i + 1) * ntheta] = q_i
        if nturbines == 1:
            k = np.array([ka])

    return _compute_residual_output(A, q, k, ntheta, w)


# ==============================================================================
# MAIN SOLVER ENTRYPOINT
# ==============================================================================

def actuatorcylinder(
    turbines: list[Turbine],
    env: Environment,
    ntheta: int,
    w0: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Solves aerodynamic performance for single or coupled VAWT rotors using Actuator Cylinder theory.

    Parameters
    ----------
    turbines : list[Turbine]
        List of instantiated `Turbine` physical objects in array layout.
    env : Environment
        Fluid environment definition.
    ntheta : int
        Azimuthal grid points count around each rotor ($2\pi$ space).
    w0 : np.ndarray, optional
        Initial guess vector for induction velocities `[u, v]`, by default None.

    Returns
    -------
    CT : np.ndarray
        Thrust Coefficients for each turbine of shape `(nturbines,)`.
    CP : np.ndarray
        Power Coefficients for each turbine of shape `(nturbines,)`.
    Rp : np.ndarray
        Radial force distributions matrix of shape `(ntheta, nturbines)`.
    Tp : np.ndarray
        Tangential force distributions matrix of shape `(ntheta, nturbines)`.
    Zp : np.ndarray
        Vertical force distributions matrix of shape `(ntheta, nturbines)`.
    theta : np.ndarray
        Azimuthal angle discretization vector of shape `(ntheta,)`.
    w_opt : np.ndarray
        Converged state vector of induction velocities.
    warnings : list[str], optional
        List of convergence warning messages (returned as an 8th element if present).
    """
    centerX = np.array([t.centerX for t in turbines])
    centerY = np.array([t.centerY for t in turbines])
    radii = np.array([t.r for t in turbines])

    Ax, Ay, theta = matrixAssemble(centerX, centerY, radii, ntheta)

    ntheta = len(theta)
    nturbines = len(turbines)
    tol = 1e-6
    solver_warnings: list[str] = []

    CT = np.zeros(nturbines)
    CP = np.zeros(nturbines)
    Rp = np.zeros((ntheta, nturbines))
    Tp = np.zeros((ntheta, nturbines))
    Zp = np.zeros((ntheta, nturbines))
    k = np.zeros(nturbines)

    # Pre-stack influence matrices outside iteration loops
    A_full = np.vstack([Ax, Ay])

    if w0 is not None and nturbines > 1:
        w0_coupled = w0
        for i in range(nturbines):
            u_start = w0[i * ntheta : (i + 1) * ntheta]
            v_start = w0[ntheta * nturbines + i * ntheta : ntheta * nturbines + (i + 1) * ntheta]
            _, k[i], *_ = radialforce(u_start, v_start, theta, turbines[i], env)
    else:
        w0_coupled = np.zeros(nturbines * ntheta * 2) if w0 is None else w0

        for i in range(nturbines):
            if w0 is not None and nturbines == 1:
                w0_single = w0
            else:
                w0_single = np.zeros(ntheta * 2)

            idx = np.arange(i * ntheta, (i + 1) * ntheta)
            A_single = np.vstack([Ax[idx][:, idx], Ay[idx][:, idx]])

            def resid_single(x: np.ndarray) -> np.ndarray:
                u = x[:ntheta]
                v = x[ntheta:]
                q_val, ka_val, *_ = radialforce(u, v, theta, turbines[i], env)
                return _compute_residual_fast_single(A_single, q_val, ka_val, x)

            result = root(resid_single, w0_single, method='lm', tol=tol)
            w_single = result.x

            if not result.success:
                msg = f"Solver failed to converge for single Turbine {i + 1}: {result.message}"
                logger.warning(msg)
                solver_warnings.append(msg)

            u = w_single[:ntheta]
            v = w_single[ntheta:]
            _, k[i], CT[i], CP[i], Rp[:, i], Tp[:, i], Zp[:, i] = radialforce(u, v, theta, turbines[i], env)

            if nturbines > 1:
                w0_coupled[i * ntheta : (i + 1) * ntheta] = u
                w0_coupled[ntheta * nturbines + i * ntheta : ntheta * nturbines + (i + 1) * ntheta] = v

        if nturbines == 1:
            if solver_warnings:
                return CT, CP, Rp, Tp, Zp, theta, w_single, solver_warnings
            return CT, CP, Rp, Tp, Zp, theta, w_single

    def resid_multiple(x: np.ndarray) -> np.ndarray:
        return residual(x, A_full, theta, k, turbines, env)

    result = root(resid_multiple, w0_coupled, method='lm', tol=tol)
    w_coupled = result.x

    if not result.success:
        msg = f"Solver failed to converge for coupled multi-turbine array: {result.message}"
        logger.warning(msg)
        solver_warnings.append(msg)

    for i in range(nturbines):
        idx = list(range(i * ntheta, (i + 1) * ntheta))
        u = w_coupled[idx]
        v = w_coupled[ntheta * nturbines + np.array(idx)]
        _, _, CT[i], CP[i], Rp[:, i], Tp[:, i], Zp[:, i] = radialforce(u, v, theta, turbines[i], env)

    if solver_warnings:
        return CT, CP, Rp, Tp, Zp, theta, w_coupled, solver_warnings

    return CT, CP, Rp, Tp, Zp, theta, w_coupled

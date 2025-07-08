import numpy as np
import math
import os
import h5py
import matplotlib
from scipy.integrate import quad
from scipy.optimize import root
from typing import Callable, Tuple
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # Define a different interactive backend
from .data_generation import get_cl_cd_neuralfoil 

# Coefficients of influence
def panelIntegration(xvec, yvec, thetavec, ifunc):
    """
    Perform panel integration to compute influence coefficients.

    This function applies for both Ay and Dx depending on the function passed.
    
    Parameters
    ----------
    xvec : ndarray
        Array of x-coordinates of the panels.
        
    yvec : ndarray
        Array of y-coordinates of the panels.
        
    thetavec : ndarray
        Array of angles (in radians) at which the integration is performed.
        
    ifunc : Callable
        The integrand function to be used for integration (either for Ay or Dx).
        
    Returns
    -------
    A : ndarray
        The result of the integration, with shape (nx, ntheta), where nx is the
        number of panels and ntheta is the number of integration points.
    """
    #Inicializar
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] # Assumes equally spaced angles
    A = np.zeros((nx, ntheta))

    for i in range(nx):
        # Redefine the function for use in integration
        def integrand(phi):
            return ifunc(xvec[i], yvec[i], phi)

        for j in range(ntheta):
            # Perform adaptive integration
            result, error = quad(
                integrand,
                thetavec[j] -dtheta / 2,
                thetavec[j] + dtheta / 2,
                epsabs=1e-10
            )
            A[i, j] = result

    return A

def Dxintegrand(x, y, phi):
    """
    Integrand function for computing Dx.

    Parameters
    ----------
    x : float
        x-coordinate of the point.
        
    y : float
        y-coordinate of the point.
        
    phi : float
        Angle of integration (in radians).
        
    Returns
    -------
    float
        The value of the integrand at the given point and angle.
    """
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)

    print(v1, v2)
    # v1 and v2 must not be zero because we never integrate self. RxII handles this situation.
    return (v1 * math.sin(phi) - v2 * math.cos(phi)) / (2 * math.pi * (v1 * v1 + v2 * v2))

def Ayintegrand(x, y, phi):
    """
    Integrand function for computing Ay.

    Parameters
    ----------
    x : float
        x-coordinate of the point.
        
    y : float
        y-coordinate of the point.
        
    phi : float
        Angle of integration (in radians).
        
    Returns
    -------
    float
        The value of the integrand at the given point and angle.
    """
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)
    if abs(v1) < 1e-12 and abs(v2) < 1e-12:
        # Occurs when integrating self; the function is symmetric around the singularity and should integrate to zero
        return 0.0
    return (v1 * math.cos(phi) + v2 * math.sin(phi)) / (2 * math.pi * (v1 * v1 + v2 * v2))

def AyIJ(xvec, yvec, thetavec):
    """
    Compute AyIJ by integrating with the Ayintegrand function.

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
    Ay : ndarray
        The result of the Ay integration for each panel.
    """
    return panelIntegration(xvec, yvec, thetavec, Ayintegrand)

def DxIJ(xvec, yvec, thetavec):
    """
    Compute DxIJ by integrating with the Dxintegrand function.

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
    Dx : ndarray
        The result of the Dx integration for each panel.
    """
    return panelIntegration(xvec, yvec, thetavec, Dxintegrand)

def WxIJ(xvec, yvec, thetavec):
    """
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
    """
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
            k = np.searchsorted(thetavec + dtheta / 2, thetak, side="right")
            if 0 <= k < ntheta:
                Wx[i, k] = -1.0
                Wx[i, ntheta - k - 1] = 1.0

    return Wx

def DxII(thetavec):
    """
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
    """
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] # Assumes equally spaced values
    Rx = (dtheta / (4 * np.pi)) * np.ones((ntheta, ntheta))

    for i in range(ntheta):
        if i < ntheta // 2:
            Rx[i, i] = (-1 + 1.0 / ntheta) / 2.0
        else:
            Rx[i, i] = (1 + 1.0 / ntheta) / 2.0

    return Rx

def WxII(thetavec):
    """
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
    """
    ntheta = len(thetavec)
    Wx = np.zeros((ntheta, ntheta))

    for i in range(ntheta // 2, ntheta):
        Wx[i, ntheta - 1 - i] = -1

    return Wx

def precomputeMatrices(ntheta, modulepath):
    """
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
    """
    dtheta = 2 * np.pi / ntheta
    theta = np.arange(dtheta / 2, 2 * np.pi, dtheta)

    # Precompute matrices
    Dxself = DxII(theta)
    Wxself = WxII(theta)
    Ayself = AyIJ(-np.sin(theta), np.cos(theta), theta)

    filepath = f'{modulepath}/theta-{ntheta}.h5'
    with h5py.File(filepath, 'w') as file:
        file.create_dataset("theta", data=theta)
        file.create_dataset("Dx", data=Dxself)
        file.create_dataset("Wx", data=Wxself)
        file.create_dataset("Ay", data=Ayself)

    return filepath

def matrixAssemble(centerX, centerY, radius, ntheta):
    """
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
    """
    file = f"theta-{ntheta}.h5"
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
    """
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
    """
    def __init__(self, r: float, chord: float, twist: float, delta: float, B: int, Omega: float, centerX: float, centerY: float):
        self.r = r
        self.chord = chord
        self.twist = twist
        self.delta = delta
        self.B = B
        self.Omega = Omega
        self.centerX = centerX
        self.centerY = centerY

class Environment:
    """
    Class representing the wind and fluid properties of the environment.

    Parameters
    ----------
    Vinf : float
        Free stream wind velocity [m/s].
    rho : float
        Air density [kg/m^3].
    mu : float
        Dynamic viscosity of the air [Pa·s].
    """
    def __init__(self, Vinf: float, rho: float, mu: float):
        self.Vinf = Vinf
        self.rho = rho
        self.mu = mu

def radialforce(uvec, vvec, thetavec, turbine: Turbine, env: Environment, config, turbine_index, airfoil_index):
    """
    Calculates aerodynamic forces and performance coefficients for a VAWT using the actuator cylinder method.

    Parameters
    ----------
    uvec : ndarray
        Axial induction factor as a function of azimuth angle (unitless).
    vvec : ndarray
        Tangential induction factor as a function of azimuth angle (unitless).
    thetavec : ndarray
        Azimuthal angle vector [rad].
    turbine : Turbine
        Turbine object containing turbine geometry and operating parameters.
    env : Environment
        Environment object containing wind speed and air properties.

    Returns
    -------
    q : ndarray
        Local force coefficient per unit length.
    ka : float
        Correction factor accounting for nonlinear effects and induction.
    CT : float
        Thrust coefficient.
    CP : float
        Power coefficient.
    Rp : ndarray
        Radial force per unit span along the azimuth [N/m].
    Tp : ndarray
        Tangential force per unit span along the azimuth [N/m].
    Zp : ndarray
        Axial force per unit span along the azimuth [N/m].

    Notes
    -----
    Two correction models are available for calculating the correction factor `ka`.
    The active model is based on a piecewise analytical expression depending on `CT`.
    An alternative model based on a fitted polynomial is also provided but commented out.
    """
    # Unpacking turbine and environment parameters
    r = turbine.r              # Rotor radius
    chord = turbine.chord      # Blade chord length
    twist = turbine.twist      
    delta = turbine.delta      
    B = turbine.B              # Number of blades
    Omega = turbine.Omega      # Rotational speed (rad/s)
    Vinf = env.Vinf            # Freestream wind speed
    rho = env.rho              # Air density

    # Direction of rotation: +1 or -1
    rotation = np.sign(Omega)

    # Normal (Vn) and tangential (Vt) components of relative velocity
    Vn = Vinf * (1.0 + uvec) * np.sin(thetavec) - Vinf * vvec * np.cos(thetavec)
    Vt = (rotation * (Vinf * (1.0 + uvec) * np.cos(thetavec) + Vinf * vvec * np.sin(thetavec)) + abs(Omega) * r)

    W = np.sqrt(Vn**2 + Vt**2) # Magnitude of relative wind velocity
    phi = np.arctan2(Vn, Vt) # Flow angle (between rotor plane and relative velocity)
    alpha = phi - turbine.twist # Angle of attack (flow angle minus blade pitch)

    # Lift and drag coefficients from airfoil function
    cl = np.zeros_like(alpha)
    cd = np.zeros_like(alpha)

    for i in range(len(alpha)):
        cl[i], cd[i] = get_cl_cd_neuralfoil(
            alpha=alpha[i],
            W=W[i],
            turbine_index=turbine_index,
            airfoil_index=airfoil_index
        )

    # Normal and tangential force coefficients in the rotor frame
    cn = cl * np.cos(phi) + cd * np.sin(phi)
    ct = cl * np.sin(phi) - cd * np.cos(phi)

    sigma = B * chord / r # Solidity (blade area / swept area)
    q = sigma / (4 * np.pi) * cn * (W / Vinf)**2 # Local thrust coefficient

    # Instantaneous forces
    qdyn = 0.5 * rho * W**2 # Dynamic pressure at each azimuthal position
    Rp = -cn * qdyn * chord # Radial (normal) force per unit span
    Tp = ct * qdyn * chord / np.cos(delta) # Tangential force per unit span (contributes to torque)
    Zp = -cn * qdyn * chord * np.tan(delta) # Axial force due to blade tilt


    # Nonlinear correction factor for induction (ka)
    integrand = (W / Vinf)**2 * (cn * np.sin(thetavec) - rotation * ct * np.cos(thetavec) / np.cos(delta))
    CT = sigma / (4 * np.pi) * np.trapz(integrand, x=thetavec)

    if CT > 2.0:
        a = 0.5 * (1.0 + np.sqrt(1.0 + CT))
        ka = 1.0 / (a - 1)
    elif CT > 0.96:
        a = 1.0 / 7 * (1 + 3.0 * np.sqrt(7.0 / 2 * CT - 3))
        ka = 18.0 * a / (7 * a**2 - 2 * a + 4)
    else:
        a = 0.5 * (1 - np.sqrt(1.0 - CT))
        ka = 1.0 / (1 - a)

    '''
    # Alternative correction factor model (uncomment to use instead)
    a=0.0892074*CT**3 + 0.0544955*CT**2 + 0.251163*CT - 0.0017077

    if a <= 0.15:
        ka = 1/(1-a)

    elif a > 0.15:
        ka = 1/(1-a)*(0.65 + 0.35*math.exp(-4.5*(a-0.15)))
    '''  

    # Power coefficient (CP)
    H = 1.0                    # Rotor height (unit length)
    Sref = 2 * r * H           # Swept area
    Q = r * Tp                 # Torque at each position
    P = abs(Omega) * B / (2 * np.pi) * np.trapz(Q, x=thetavec)  # Total power
    CP = P / (0.5 * rho * Vinf**3 * Sref)  # Power coefficient

    return q, ka, CT, CP, Rp, Tp, Zp


#------------------------------------
#
#-------- solve the system --------

def residual(w, A, theta, turbine, env, config, turbine_index, airfoil_index):
    """
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
    """
    # Initial configuration
    ntheta = len(theta)

    # split u and v
    u = w[:ntheta]
    v = w[ntheta:]

    # Compute radial force, returns q (length ntheta) and the scalar kappa (ka)
    q, ka, *_ = radialforce(u, v, theta, turbine, env, config, turbine_index, airfoil_index)

    # Build k_mult twice (for u- and v-equations)
    kmult = np.full(2 * ntheta, ka)

    return (A @ q) * kmult - w

def actuatorcylinder(turbine, env, ntheta, config, turbine_index, airfoil_index):
    """
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
    """
    # Set turbine geometry
    centerX = turbine.centerX
    centerY = turbine.centerY
    radius = turbine.r

    # Assemble global matrices
    Ax, Ay, theta = matrixAssemble(centerX, centerY, radius, ntheta)
    A = np.vstack([Ax, Ay])

    w0 = np.zeros(2 * len(theta))

    sol = root(residual, w0, args=(A, theta, turbine, env, config, turbine_index, airfoil_index), tol=1e-6)
    if not sol.success:
        raise RuntimeError(f'Solver did not converge: {sol.message}')

    w = sol.x
    u, v = w[:len(theta)], w[len(theta):]

    q, ka, CT, CP, Rp, Tp, Zp = radialforce(u, v, theta, turbine, env, config, turbine_index, airfoil_index)
    
    return CT, CP, Rp, Tp, Zp, theta


#------------------------------------
#-------- Auxiliary Methods --------

def trapz(x, y):
    """
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
    """
    integral = 0.0
    for i in range(len(x) - 1):
        integral += (x[i+1] - x[i]) * 0.5 * (y[i] + y[i+1])
    return integral

def pInt(theta, f):
    """
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
    """
    # Compute the integral using the trapezoidal rule
    integral = trapz(theta, f)

    # Add the contribution from the periodic boundary points
    dtheta = 2 * theta[0] # Assume equal spacing, starts at 0
    integral += dtheta * 0.5 * (f[0] + f[-1])
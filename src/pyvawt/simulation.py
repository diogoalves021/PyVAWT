import numpy as np
import math
import os
import h5py
import csv
import time
import traceback
import copy
from scipy.integrate import quad
from scipy.optimize import root
import matplotlib.pyplot as plt

from src.pyvawt.submodels.flow_curvature import FlowCurvatureManager, FlowCurvatureModel
from .data_reading import readaerodyn
from .utils import save_config, get_cl_cd


# Coefficients of influence
def panelIntegration(xvec, yvec, thetavec, ifunc):
    '''
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
    '''
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
    '''
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
    '''
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)

    print(v1, v2)
    # v1 and v2 must not be zero because we never integrate self. RxII handles this situation.
    return (v1 * math.sin(phi) - v2 * math.cos(phi)) / (2 * math.pi * (v1 * v1 + v2 * v2))

def Ayintegrand(x, y, phi):
    '''
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
    '''
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)
    if abs(v1) < 1e-12 and abs(v2) < 1e-12:
        # Occurs when integrating self; the function is symmetric around the singularity and should integrate to zero
        return 0.0
    return (v1 * math.cos(phi) + v2 * math.sin(phi)) / (2 * math.pi * (v1 * v1 + v2 * v2))

def AyIJ(xvec, yvec, thetavec):
    '''
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
    '''
    return panelIntegration(xvec, yvec, thetavec, Ayintegrand)

def DxIJ(xvec, yvec, thetavec):
    '''
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

def radialforce(uvec, vvec, thetavec, turbine: Turbine, env: Environment, config, turbine_index, airfoil_index, flow_manager):
    '''
    Calculate aerodynamic forces and performance coefficients for a Vertical Axis Wind Turbine (VAWT)
    using the actuator cylinder method.

    This function computes aerodynamic forces based on the specified method for obtaining
    lift (Cl) and drag (Cd) coefficients:
    
    - "neuralfoil": Uses a neural network model to predict Cl and Cd.
    - "file": Uses precomputed airfoil polars read from a file and interpolated.

    Parameters
    ----------
    uvec : ndarray
        Axial induction factor as a function of azimuthal angle (unitless).
    vvec : ndarray
        Tangential induction factor as a function of azimuthal angle (unitless).
    thetavec : ndarray
        Azimuthal angle vector [rad].
    turbine : Turbine
        Turbine object containing geometry and operational parameters.
    env : Environment
        Environment object containing wind speed and air properties.
    config : dict
        Simulation configuration dictionary, including aerodynamic method selection.
    turbine_index : int
        Turbine index (used for multi-turbine simulations).
    airfoil_index : int
        Airfoil index (used with the "neuralfoil" method).

    Returns
    -------
    q : ndarray
        Local force coefficient per unit length.
    ka : float
        Correction factor for nonlinear effects and induction.
    CT : float
        Thrust coefficient.
    CP : float
        Power coefficient.
    Rp : ndarray
        Radial (normal) force per unit span along the azimuth [N/m].
    Tp : ndarray
        Tangential force per unit span along the azimuth [N/m].
    Zp : ndarray
        Axial force per unit span along the azimuth [N/m].

    Notes
    -----
    - The aerodynamic coefficient calculation method is defined by ``config['aero']['method']``.
    - Two correction models for the induction factor ``ka`` are implemented: a piecewise analytical model (active)
      and an alternative polynomial fit (commented out for reference).
    '''
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
    if flow_manager is None:
        alpha_corr = alpha
    else:
        alpha_corr = flow_manager.corrected_flow(alpha, Omega, W)
    alpha = alpha_corr

    # Lift and drag coefficients from airfoil function
    cl = np.zeros_like(alpha)
    cd = np.zeros_like(alpha)

    # Calls neuralfoil or interpolate aero data
    cl, cd = get_cl_cd(alpha, W, turbine_index, airfoil_index, config)

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

def residual(w, A, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager):
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
    q, ka, *_ = radialforce(u, v, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager)

    # Build k_mult twice (for u- and v-equations)
    kmult = np.full(2 * ntheta, ka)

    return (A @ q) * kmult - w

def actuatorcylinder(turbine, env, ntheta, config, turbine_index, airfoil_index, flow_manager):
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

    sol = root(residual, w0, args=(A, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager), tol=1e-6)
    if not sol.success:
        raise RuntimeError(f'Solver did not converge: {sol.message}')

    w = sol.x
    u, v = w[:len(theta)], w[len(theta):]

    q, ka, CT, CP, Rp, Tp, Zp = radialforce(u, v, theta, turbine, env, config, turbine_index, airfoil_index, flow_manager)
    
    return CT, CP, Rp, Tp, Zp, theta

def get_radius_from_config(turbine_config: dict) -> float:
    '''
    Computes the rotor radius of a vertical-axis wind turbine based on the configuration.

    The radius can be defined in two ways:
    - 'manual': uses the provided value of 'r' directly.
    - 'auto': calculates the radius from other parameters using the formula:
        r = (B * chord) / (solidity * H)

    Parameters
    ----------
    turbine_config : dict
        Dictionary containing turbine parameters. Must include:
        - radius_mode : str
            Either 'manual' or 'auto'.
        - If 'manual': must provide 'r'.
        - If 'auto' : must provide 'B', 'chord', 'solidity', and 'H'.

    Returns
    -------
    float
        The computed or retrieved rotor radius.

    Raises
    ------
    ValueError
        If required parameters are missing or if an invalid mode is specified.
    '''
    mode = turbine_config.get('radius_mode', 'manual')

    if mode == 'manual':
        r = turbine_config.get('r', None)
        if r is None:
            raise ValueError("Mode 'manual' selected, but 'r' was not provided.")
        return r

    elif mode == 'auto':
        H = turbine_config.get('H', None)
        if H is None:
            raise ValueError("Mode 'auto' selected, but 'H' was not provided.")
        B = turbine_config.get('B')
        chord = turbine_config.get('chord')
        solidity = turbine_config.get('solidity')
        if None in (B, chord, solidity):
            raise ValueError('Missing parameters for automatic radius calculation.')
        return (B * chord) / (solidity * H)

    else:
        raise ValueError(f'Invalid radius mode: {mode}. Use "manual" or "auto".')


def initialize_turbine_and_environment(config):
    '''
    Initializes the turbine and environment objects based on the configuration file.

    Parameters
    ----------
    config : dict
        Dictionary containing simulation, turbine, and environment parameters.

    Returns
    -------
    turbine : Turbine
        The initialized Turbine object.
    env : Environment
        The Environment object initialized with freestream conditions.
    simulation_params : dict
        Dictionary with general simulation parameters.
    turbine_params : dict
        Dictionary with turbine-specific parameters.
    environment_params : dict
        Dictionary with environmental parameters.
    r : float
        Rotor radius (in meters).
    ntheta : int
        Number of azimuthal discretization points.

    Notes
    -----
    The function also reads airfoil data using the `readaerodyn_neuralfoil` function
    and uses it to initialize the turbine's aerodynamic properties.
    '''
    turbine_params = config['turbine']
    environment_params = config['environment']
    simulation_params = config['simulation']

    r = get_radius_from_config(turbine_params)
    config['turbine']['r'] = r

    twist = turbine_params['twist']
    delta = turbine_params['delta']
    chord = turbine_params['chord']
    B = turbine_params['B']
    solidity = turbine_params['solidity']
    centerX = turbine_params['centerX']
    centerY = turbine_params['centerY']
    Omega = turbine_params['Omega']
    ntheta = turbine_params['ntheta']

    Vinf = environment_params['Vinf']
    rho = environment_params['rho']
    mu = environment_params['mu']

    turbine = Turbine(r, chord, twist, delta, B, Omega, centerX, centerY)
    env = Environment(Vinf, rho, mu)

    aero_params = config.get('aero', {})
    method = aero_params.get('method', 'neuralfoil')
    if method == 'file':
        af_func = readaerodyn(aero_params['file'])
        config['aero']['af_func'] = af_func
    else:
        config['aero']['af_func'] = None

    return turbine, env, simulation_params, turbine_params, environment_params, r, ntheta

def run_simulation_case(params, base_config, flow_cfg=None):
    '''
    Executes a single aerodynamic simulation for a vertical-axis wind turbine (VAWT)
    using the provided parameters and base configuration.

    Parameters
    ----------
    params : tuple
        A tuple containing the following:
        - airfoil_index : int
            Index of the airfoil name in the airfoil list from the configuration.
        - turbine_index : int
            Index of the turbine (reserved for future multi-turbine support).
        - chord : float
            Blade chord length in meters.
        - solidity : float
            Turbine solidity (dimensionless).
        - vinf : float
            Freestream wind velocity in m/s.

    base_config : dict
        Base configuration dictionary loaded from YAML, containing all simulation settings.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - 'name' : str
            Name of the folder used to save results.
        - 'status' : str
            'OK' if successful, or an error message otherwise.
        - 'time_sec' : float
            Duration of the simulation in seconds.
        - 'traceback' : str, optional
            Traceback info included if the simulation fails.

    Notes
    -----
    - The simulation evaluates turbine performance across a range of TSR (Tip-Speed Ratio).
    - The parameter `fixed_parameter` in the config determines which quantity is held constant:
        * 'vinf': wind speed is fixed, Omega is varied.
        * 'omega': angular velocity is fixed, wind speed is varied.
    - Simulation results include thrust/torque/power coefficients, optionally saved as .dat/.csv and plots.
    - Output files are saved to: src/results/temporary_results/<case_name>
    - Designed for batch processing; assumes one turbine per run.
    '''
    airfoil_index, turbine_index, chord, solidity, vinf = params
    config = copy.deepcopy(base_config)  # Deep copy

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

    airfoil_name = config['simulation']['airfoil'][airfoil_index]
    config['simulation']['airfoil'] = airfoil_name
    config['turbine']['chord'] = chord
    config['turbine']['solidity'] = solidity
    config['environment']['Vinf'] = vinf
    angular_velocity = config['turbine']['Omega']
    delta = config['turbine']['delta']
    r = config['turbine']['r']

    if flow_cfg is None:
        flow_cfg = {}

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
        save_config(config, os.path.join(result_dir, 'config_used.yaml'))


    turbine, env, sim_params, _, _, _, ntheta = initialize_turbine_and_environment(config)
    fixed_parameter = sim_params['fixed_parameter']
    num_turbines = sim_params['num_turbines']

    start_time = time.time()
    try:
        print(f'Simulating: {folder_name}')
        n = 20
        tsrvec = np.linspace(1, 7, n)
        CPvec = np.zeros(n)
        CTvec = np.zeros(n)
        Rpvec = np.zeros(n)
        Tpvec = np.zeros(n)
        Zpvec = np.zeros(n)

        if fixed_parameter == 'vinf':
            # Vinf is fixed thus omega is not constant for each tsr
            for i, tsr in enumerate(tsrvec):
                turbine.Omega = vinf * tsr / r
                CT, CP, Rp, Tp, Zp, _ = actuatorcylinder(
                    turbine, env, ntheta, config, turbine_index, airfoil_index, flow_manager
                )
                CPvec[i], CTvec[i], Rpvec[i], Tpvec[i], Zpvec[i] = (
                    CP,
                    CT,
                    Rp[0],
                    Tp[0],
                    Zp[0],
                )

        elif fixed_parameter == 'omega':
            # Omega is fixed thus vinf is not constant for each tsr
            for i, tsr in enumerate(tsrvec):
                turbine.Omega = angular_velocity
                env.Vinf = turbine.Omega * r / tsr
                CT, CP, Rp, Tp, Zp, _ = actuatorcylinder(
                    turbine, env, ntheta, config, turbine_index, airfoil_index, flow_manager
                )
                CPvec[i], CTvec[i], Rpvec[i], Tpvec[i], Zpvec[i] = (
                    CP,
                    CT,
                    Rp[0],
                    Tp[0],
                    Zp[0],
                )

        else:
            raise ValueError("Invalid value for 'fixed_parameter'. Use 'vinf' or 'omega'.")

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


        plt.figure()
        plt.plot(tsrvec, CPvec)
        plt.title(f'$C_p$ x TSR - {airfoil_name}')
        plt.xlabel('TSR')
        plt.ylabel('$C_p$')
        plt.grid(True)
        plt.tight_layout()
        if save_plot:
            plot_filename = f'cp_curve_{airfoil_name}.{image_format}'
            plt.savefig(os.path.join(result_dir, plot_filename), format=image_format, dpi=dpi)

        plt.close()

        elapsed = time.time() - start_time
        return {'name': folder_name, 'status': 'OK', 'time_sec': round(elapsed, 2)}

    except Exception as e:
        return {
            'name': folder_name,
            'status': f'ERROR: {e}',
            'time_sec': round(time.time() - start_time, 2),
            'traceback': traceback.format_exc(limit=2),
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
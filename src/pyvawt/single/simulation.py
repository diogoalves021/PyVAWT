from numba import jit
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
from src.pyvawt.submodels.boeing_vertol import af, Boeing_Vertol
from src.pyvawt.single.data_reading import readaerodyn
from src.pyvawt.single.utils import save_config, format_time
from src.pyvawt.single.data_generation import get_cl_cd_neuralfoil
from src.pyvawt.single.utils import load_config, get_tc_from_airfoil, detect_stall_angles, save_config

# Coefficients of influence
# @jit
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
@jit
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
@jit
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
# @jit
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
# @jit
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
@jit
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
@jit
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
@jit
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
    
class NeuralFoilAerodynamics(Aerodynamics):
    '''
    Aerodynamics model using a neural network (NeuralFoil) to calculate Cl and Cd.

    Parameters
    ----------
    turbine_index : int
        Turbine index used to access neural network training data.
    airfoil_index : int
        Airfoil index.
    '''
    def __init__(self,  turbine_index, airfoil_index):
        self.turbine_index = turbine_index
        self.airfoil_index = airfoil_index

    def get_cl_cd(self, alpha, W):
        '''
        Returns Cl and Cd using the neural network model.

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].
        W : float
            Relative wind speed [m/s].

        Returns
        -------
        tuple of floats
            (Cl, Cd)
        '''
        return get_cl_cd_neuralfoil(alpha, W, self.turbine_index, self.airfoil_index)
    
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

def fast_trapz(y, x):
    """
    Integração trapezoidal 1D ultra-rápida.
    Evita todo o overhead de validação de eixos e dimensões do scipy/numpy.
    Utiliza np.dot para calcular o acumulado diretamente em C.
    """
    return 0.5 * np.dot(y[:-1] + y[1:], np.diff(x))


def radialforce(uvec, vvec, thetavec, turbine: Turbine, env: Environment,
                config, turbine_index, airfoil_index, flow_manager, z=None, H=None):
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
    solidity = turbine.solidity
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
 
    # Lift and drag coefficients from airfoil function
    cl = np.zeros_like(alpha)
    cd = np.zeros_like(alpha)
    cm = np.zeros_like(alpha)

    aero_cfg = config.get('solver', {}).get('aero', config.get('aero', {}))
    use_dynamic_stall = aero_cfg.get('dynamic_stall', True)
    
    sim3d_cfg = config.get('solver', {}).get('simulation3d', config.get('simulation', {}).get('simulation3d', {}))
    use_tip_loss = sim3d_cfg.get('tip_loss', False)


    if use_dynamic_stall:

        # Angular derivative
        # d(alpha)/d(theta) 
        dalpha_dtheta = np.gradient(alpha, thetavec, edge_order=2)
        
        # d(alpha)/dt
        adot = Omega * dalpha_dtheta

        # Normalized alpha
        adotnorm = adot * chord / (2.0 * W)

        # Local Reynolds
        Re = rho * W * chord / env.mu

        # Local mach 
        v_sound = 340.0
        umach = W / v_sound

        # Uses boeing-Vertol dynamic stall correction
        flagL = int(env.BV_DynamicFlagL)
        flagD = int(env.BV_DynamicFlagD)
        
        airfoil_list = config.get('solver', {}).get('neuralfoil', {}).get('airfoil', config.get('simulation', {}).get('airfoil', []))
        airfoil_name = airfoil_list[airfoil_index]
        tc = get_tc_from_airfoil(airfoil_name)

        # Calls neuralfoil or interpolate aero data
        cl_static, cd_static = turbine.aero.get_cl_cd(alpha, W)
        cm_static = np.zeros_like(cl_static)

        # Static AOA
        aoaStallPos = turbine.aero.aoaStallPos
        aoaStallNeg = turbine.aero.aoaStallNeg
        
        for i, a in enumerate(alpha):
            try:
                cl[i], cd[i], cm[i], flagL, flagD = Boeing_Vertol(
                    cl_static[i],
                    cd_static[i],
                    cm_static[i],
                    a,
                    adotnorm[i],
                    umach[i],
                    Re[i],
                    aoaStallPos,
                    aoaStallNeg,
                    0.0,       # AOA0
                    tc,
                    flagL,
                    flagD,
                    turbine,
                    env,
                    turbine_index,
                    airfoil_index,
                )
            except Exception as e:
                # Professional debug message
                error_msg = (
                    f"\n[ERROR] Boeing-Vertol calculation failed\n"
                    f"Index: {i}\n"
                    f"Alpha: {a:.6f} rad\n"
                    f"Reynolds: {Re[i]:.2f}\n"
                    f"Thickness/Chord: {tc:.2f}\n"
                    f"Airfoil index: {airfoil_index}\n"
                    f"Total alpha points: {len(alpha)}\n"
                )
                print(error_msg)
                raise RuntimeError(error_msg) from e

        env.BV_DynamicFlagL = flagL
        env.BV_DynamicFlagD = flagD

    else:
        # Calls neuralfoil or interpolate aero data
        cl, cd = turbine.aero.get_cl_cd(alpha, W)

    # Normal and tangential force coefficients in the rotor frame
    cn = cl * np.cos(phi) + cd * np.sin(phi)
    ct = cl * np.sin(phi) - cd * np.cos(phi)
    
    if use_tip_loss and z is not None and H is not None:
        cn_corr, ct_corr, F = apply_tip_loss(cn, ct, z, H, turbine, env, Vinf, uvec)
        
        F_values = []
        F_values.append(F)

        print(
            "Tip-loss:",
            f"min={np.min(F_values):.3f}",
            f"mean={np.mean(F_values):.3f}",
            f"max={np.max(F_values):.3f}",
        )

    else:
        cn_corr = cn
        ct_corr = ct 
  
    # Automatically check if solidity is based on diameter or radius and use correct formulation
    sigma_r = B * chord / r          # baseada no raio
    sigma_D = B * chord / (2 * r)    # baseada no diâmetro

    if abs(solidity - sigma_r) < abs(solidity - sigma_D):
        sigma = sigma_r
        # print('Solidez baseada no raio!')
    else:
        sigma = 2 * solidity  # converte para definição baseada no raio
        # print('Solidez baseada no diâmetro!')

    q = sigma / (4 * np.pi) * cn * (W / Vinf)**2 # Local thrust coefficient

    # Instantaneous forces
    qdyn = 0.5 * rho * W**2 # Dynamic pressure at each azimuthal position
    Rp = -cn_corr * qdyn * chord # Radial (normal) force per unit span
    Tp = ct_corr * qdyn * chord / np.cos(delta) # Tangential force per unit span (contributes to torque)
    Zp = -cn_corr * qdyn * chord * np.tan(delta) # Axial force due to blade tilt


    # Nonlinear correction factor for induction (ka)
    integrand = (W / Vinf)**2 * (cn * np.sin(thetavec) - rotation * ct * np.cos(thetavec) / np.cos(delta))
    CT = sigma / (4 * np.pi) * fast_trapz(integrand, thetavec)

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
    #'''  

    # Power coefficient (CP)
    Href = 1.0                    # Rotor height (unit length)
    Sref = 2 * r * Href           # Swept area
    Q = r * Tp                 # Torque at each position
    P = abs(Omega) * B / (2 * np.pi) * fast_trapz(Q, thetavec)  # Total power
    CP = P / (0.5 * rho * Vinf**3 * Sref)  # Power coefficient

    return q, ka, CT, CP, Rp, Tp, Zp, alpha, W

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
    simulation_params = config['solver']

    # Remove dependency on get_radius_from_config
    r = turbine_params['r']

    twist = turbine_params['twist']
    delta = turbine_params['delta']
    
    chord = np.array(turbine_params['chord'])
    B = turbine_params['B']
    solidity = np.array(turbine_params['solidity'])
    
    centerX = turbine_params['centerX']
    centerY = turbine_params['centerY']
    Omega = turbine_params['Omega']
    ntheta = turbine_params['ntheta']

    Vinf = np.array(environment_params['Vinf'])
    rho = environment_params['rho']
    mu = environment_params['mu']

    turbine = Turbine(r, chord, twist, delta, B, Omega, centerX, centerY, solidity)
    env = Environment(Vinf, rho, mu)

    solver_params = config.get('solver', {})
    method = solver_params.get('method', 'neuralfoil')
    
    if method == 'file':
        # Safely extract the nested 'path' parameter from 'file'
        file_cfg = solver_params.get('file', {})
        filename = file_cfg.get('path') if isinstance(file_cfg, dict) else None
        
        if not filename:
            raise ValueError("solver.file.path is not defined in config, required when method='file'")
        
        turbine.aero = FileAerodynamics(filename)
    else:
        turbine.aero = None

    return turbine, env, simulation_params, turbine_params, environment_params, r, ntheta

def run_simulation_case(params, base_config, flow_cfg=None, stall_angles=None, z=None, H=None):
    """
    Executes a single aerodynamic simulation sweep for a Vertical-Axis Wind Turbine (VAWT).

    This function sets up a specific turbine case based on input geometric and operational
    parameters, configures optional submodels (such as Flow Curvature), sweeps across a range
    of Tip Speed Ratios (TSR), and solves the rotor flow field using the Actuator Cylinder method.
    The resulting performance curves ($C_p$ and $C_t$ vs. TSR) and azimuthal distribution
    data are exported to disk according to the configuration settings.

    Args:
        params (tuple): Case parameters ordered as:
            `(airfoil_index, turbine_index, chord, solidity, vinf)` where:
                - airfoil_index (int): Index of the airfoil profile to evaluate.
                - turbine_index (int): Index of the turbine configuration.
                - chord (float): Blade chord length [m].
                - solidity (float): Rotor solidity ratio ($\sigma = B c / R$).
                - vinf (float): Free-stream wind velocity [m/s].
        base_config (dict): Deep-copyable base configuration dictionary containing 
            'turbine', 'environment', 'solver', 'submodels', and 'output' sections.
        flow_cfg (dict, optional): Flow curvature configuration override. Defaults to None.
        stall_angles (list of tuple, optional): Precomputed positive and negative stall 
            angles `(aoaStallPos, aoaStallNeg)` per airfoil index. Required. Defaults to None.
        z (float, optional): Vertical slice coordinate [m] for 3D multi-slice runs. Defaults to None.
        H (float, optional): Total rotor height [m] for 3D multi-slice runs. Defaults to None.

    Returns:
        dict: Simulation output dictionary containing execution status, performance metrics,
              and calculated force coefficients or error tracebacks.

    Raises:
        ValueError: If `stall_angles` is not provided.
        ValueError: If `config['solver']['fixed_parameter']` is not 'vinf' or 'omega'.
    """

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
    # Check whether the current case is running under a 3D multi-slice domain
    is_3d_mode = config.get('simulation3d', {}).get('enabled', False) or (z is not None)
    cp_theta_cfg = output_cfg.get('cp_theta', {})
    cp_theta_requested = cp_theta_cfg.get('enabled', False)
    
    # BUSINESS RULE: Cp(theta) extraction is strictly restricted to 2D simulations
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
    angular_velocity = config['turbine']['Omega']
    delta = config['turbine']['delta']
    r = config['turbine']['r']

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

    # Helper function for decimal formatting in string identifiers
    def fmt(val):
        return str(val).replace('.', 'p')

    # Construct unique folder identifier for the case outputs
    folder_name = (
        f'{airfoil_name}_ch{fmt(chord)}_sol{fmt(solidity)}_vinf{fmt(vinf)}'
        f'_delta{fmt(delta)}_r{fmt(r)}'
    )
    result_dir = os.path.join('src/results/temporary_results', folder_name)
    config['output']['result_folder'] = folder_name

    # Create destination directory if any saving option is enabled
    if save_results or save_config_used or save_plot:
        os.makedirs(result_dir, exist_ok=True)
        print(f"--> [DEBUG] Output path: {os.path.abspath(result_dir)}")

    # Archive active configuration settings
    if save_config_used:
        from src.pyvawt.single.utils import save_config
        save_config(config, os.path.join(result_dir, 'config_used.yaml'))

    # Helper function to ensure scalar conversion
    def _to_scalar(val):
        try:
            return val[0]
        except (TypeError, IndexError):
            return val

    # Instantiate core turbine and environmental objects
    turbine, env, _, _, _, _, _ = initialize_turbine_and_environment(config)
    
    fixed_parameter = config['solver']['fixed_parameter']
    ntheta = config['turbine']['ntheta']
    aero_method = config.get('solver', {}).get('method', 'neuralfoil')

    if aero_method == 'neuralfoil':
        turbine.aero = NeuralFoilAerodynamics(turbine_index=turbine_index, airfoil_index=airfoil_index)

    if stall_angles is None:
        raise ValueError('Stall angles must be provided')

    aoaStallPos, aoaStallNeg = stall_angles[airfoil_index]
    turbine.aero.aoaStallPos = aoaStallPos
    turbine.aero.aoaStallNeg = aoaStallNeg

    start_time = time.time()
    try:
        print(f'Simulating: {folder_name}')

        # TSR SWEEP VECTOR DISCRETIZATION
        tsr_cfg = config.get('solver', {}).get('tsr', {})
        tsr_min = float(tsr_cfg.get('min', 1.0))
        tsr_max = float(tsr_cfg.get('max', 7.0))
        n = int(tsr_cfg.get('n_points', 20))
        tsrvec = np.linspace(tsr_min, tsr_max, n)
        
        CPvec, CTvec = np.zeros(n), np.zeros(n)
        Rpvec, Tpvec, Zpvec = np.zeros(n), np.zeros(n), np.zeros(n)

        # TARGET TSR SELECTION
        # Automatically identify the array index corresponding to the nearest TSR
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

        # AERODYNAMIC SOLVER SWEEP LOOP
        for i, tsr in enumerate(tsrvec):
            # Adjust kinematic parameter according to operational constraint
            if fixed_parameter == 'vinf':
                turbine.Omega = vinf * tsr / _to_scalar(r)
            elif fixed_parameter == 'omega':
                turbine.Omega = _to_scalar(angular_velocity)
                env.Vinf = turbine.Omega * _to_scalar(r) / tsr
            else:
                raise ValueError("Invalid value for 'fixed_parameter'. Use 'vinf' or 'omega'.")

            # Execute Actuator Cylinder aerodynamic solver
            CT, CP, Rp, Tp_raw, Zp, theta = actuatorcylinder(
                turbine, env, ntheta, config, turbine_index, airfoil_index, flow_manager, z, H
            )

            CPvec[i], CTvec[i] = CP, CT
            Rpvec[i], Tpvec[i], Zpvec[i] = _to_scalar(Rp), _to_scalar(Tp_raw), _to_scalar(Zp)

            # AZIMUTHAL CP(THETA) EVALUATION
            # Evaluated exclusively at the index nearest to target_tsr
            if cp_theta_enabled and i == target_idx:
                theta_deg = np.degrees(theta)
                Href = 1.0
                Sref = 2 * _to_scalar(turbine.r) * Href

                # Compute local power coefficient from tangential force distribution (Tp_raw)
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

        # GLOBAL RESULTS EXPORT (Cp vs. TSR)
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

        # FIGURE GENERATION AND EXPORT
        # Primary plot: Performance curve (Cp vs. TSR)
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

        # Secondary plot: Azimuthal power coefficient (Cp vs. Theta)
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

        elapsed = time.time() - start_time
        return {
            'name': folder_name, 'status': 'OK', 'time_sec': round(elapsed, 2),
            'tsr': tsrvec, 'CP': CPvec, 'CT': CTvec, 'Tp': Tpvec, 'Rp': Rpvec, 'Zp': Zpvec
        }

    except Exception as e:
        print(f"\n[SIMULATION FAILURE]: {folder_name}")
        traceback.print_exc()
        print("-" * 60)
        
        return {
            'name': folder_name, 'status': f'ERROR: {e}',
            'time_sec': round(time.time() - start_time, 2),
            'traceback': traceback.format_exc(limit=2),
        }

def simulate_3D_turbine(base_config, stall_angles):
    """
    Executes a 3D multi-slice aerodynamic simulation for a Vertical-Axis Wind Turbine (VAWT).

    This wrapper function slices the turbine vertically along its height ($H$) based on the
    configuration parameters. It models atmospheric boundary layer wind shear profiles
    (e.g., power law or constant) using a power-law vertical discretization ($\beta$),
    runs 2D Actuator Cylinder cases for each height slice, aggregates the generated aerodynamic
    power, and computes the global 3D power coefficient ($C_{p,3D}$).

    If 3D simulation is disabled in the configuration (`solver.simulation3d.enabled: false`),
    the function falls back to executing a standard single 2D simulation case.

    Args:
        base_config (dict): Base configuration dictionary containing turbine, environment,
            and solver settings (including the 'simulation3d' section).
        stall_angles (list of tuple): Precomputed positive and negative stall angles
            `(aoaStallPos, aoaStallNeg)` per airfoil index required by the aerodynamic solver.

    Returns:
        None: The function directly exports the integrated 3D performance data 
            (`results_3D.dat`), saved plot figures (`cp_curve_3D.png`), and the copied 
            YAML configuration file (`config_used.yaml`) to the designated 3D results directory.

    Raises:
        ValueError: If an unknown `velocity_profile` string is provided in `base_config`
            (supported options: 'power_law', 'constant').
    """
    start_time_3d = time.time()

    total_CP = None
    power_total = None

    config_no_output = copy.deepcopy(base_config)
    config_no_output['output']['save'] = False
    config_no_output['output']['save_plot'] = False
    config_no_output['output']['save_config'] = False

    # AJUSTE CONFIG: Nova raiz do bloco 3D dentro de 'solver'
    sim3d_cfg = base_config.get('solver', {}).get('simulation3d', {})
    sim3d_settings = sim3d_cfg.get('settings', {})
    
    # Checa se simulação 3D está habilitada
    if not sim3d_cfg.get('enabled', False):
        print("Simulação 3D desabilitada. Rodando simulação 2D padrão...")
        # Pega parâmetros padrão do YAML
        airfoil_index = 0
        turbine_index = 0
        chord = base_config['turbine']['chord'][0]
        solidity = base_config['turbine']['solidity'][0]
        vinf = base_config['environment']['Vinf'][0]
        run_simulation_case(
            params=(airfoil_index, turbine_index, chord, solidity, vinf),
            base_config=base_config,
            stall_angles=stall_angles
        )
        return

    # AJUSTE CONFIG: Extração de parâmetros usando a nova subchave 'settings' e 'vertical_layers'
    height = sim3d_settings.get('height', 20.0)
    n_slices = sim3d_settings.get('vertical_layers', 20)
    velocity_profile = sim3d_settings.get('velocity_profile', 'constant')

    airfoil_index = sim3d_settings.get('airfoil_index', 0)
    turbine_index = sim3d_settings.get('turbine_index', 0)
    chord = base_config['turbine']['chord'][0]
    solidity = base_config['turbine']['solidity'][0]
    velocity_profile = sim3d_settings.get('velocity_profile', 'power_law')
    # dz = height / n_slices
    
    folder_name_3D = f"3D_H{height}_Ns{n_slices}"
    result_dir_3D = os.path.join('src/results/results_3D', folder_name_3D)
    os.makedirs(result_dir_3D, exist_ok=True)
    
    tsrvec_global = None
   
    rho = base_config['environment']['rho']
    r = base_config['turbine']['r']

    Vr = base_config['environment']['Vinf'][0]
    Zr = height / 2
    
    alpha = 0.13 # [0.11, 0.15] 
    
    beta = sim3d_settings.get('discretization_power', 2.0) #b = 1: linear discretization; b = 2: power_law discretization
    print(f'Beta: {beta}')
    eta = np.linspace(0, 1, n_slices + 1)
    z_nodes = height * (1 - (1 - eta)**beta)
    z_centers = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    dz_array = z_nodes[1:] - z_nodes[:-1]

    for i in range(n_slices):
        slice_start = time.time()
        
        z = z_centers[i]
        dz = dz_array[i]

        # z = dz / 2 + i * dz # Distribuição igual de slices ao longo da altura da turbina
        if velocity_profile == 'power_law':
            vinf = Vr * (z / Zr) ** alpha # velocity profile
        elif velocity_profile == 'constant':
            vinf = Vr 
            print("PERFIL DE VELOCIDADE CONSTANTE!")
        else:
            raise ValueError(f'Unknown velocity profile: {velocity_profile}')

        z_centered = z - height / 2

        print(f"Simulating slice {i+1}/{n_slices} at z={z:.2f} m")

        result = run_simulation_case(
            params=(airfoil_index, turbine_index, chord, solidity, vinf),
            base_config=config_no_output,
            stall_angles=stall_angles,
            z=z_centered,
            H=height
        )
        
        print(i, result.get("status"))
        print("CP:", result.get("CP"))

        slice_time = time.time() - slice_start
        avg_time = (time.time() - start_time_3d) / (i + 1)
        remaining = avg_time * (n_slices - (i + 1))
        print(f"Time slice {i+1}: {format_time(slice_time)} | ETA: {format_time(remaining)}")

        if result['status'] != 'OK':
            print(f"Error at slice {i}: {result['status']}")
            continue

        CP = np.array(result['CP']) 

        A_slice = 2 * r * dz

        power_slice = CP * (0.5 * rho * vinf**3 * A_slice)

        if power_total is None:
            power_total = power_slice.copy()
            tsrvec_global = np.array(result['tsr'])
        else:
            power_total += power_slice

        '''if total_CP is None:
            total_CP = CP.copy()
            tsrvec_global = np.array(result['tsr'])
        else:
            total_CP += CP '''

    print(f'Time slice {i+1}: {format_time(slice_time)}')
    # Power and Cp 3d
    # Cp_3D = total_CP / n_slices 
    A_total = 2 * r * height
    P_available = 0.5 * rho * Vr**3 * A_total

    Cp_3D = power_total / P_available

    print("[DEBUG] Shape TSR:", tsrvec_global.shape)

    return {
        'tsr': tsrvec_global,
        'cp_3d': Cp_3D,
        'result_dir': result_dir_3D,
        'elapsed_time': time.time() - start_time_3d
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

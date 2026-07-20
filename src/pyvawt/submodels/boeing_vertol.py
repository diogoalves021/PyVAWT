from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from src.pyvawt.single.data_generation import get_cl_cd_neuralfoil, load_config
#from src.pyvawt.simulation import Turbine, Environment

@jit
def abs_smooth(x, eps=1e-4):
    return np.sqrt(x*x + eps*eps)
@jit
def ksmin(values, k=300.0):
    values = np.array(values)
    return -np.log(np.sum(np.exp(-k * values))) / k
@jit
def ksmax(values, k=300.0):
    values = np.array(values)
    return np.log(np.sum(np.exp(k * values))) / k

def af(alpha, Re, Mach, turbine, env, turbine_index, airfoil_index, family_factor=None):

    rho = env.rho
    mu = env.mu
    chord = turbine.chord

    W = Re * mu / (rho * chord)
    CL, CD = get_cl_cd_neuralfoil(alpha, W, turbine_index, airfoil_index)

    CM = 0.0
    return CL, CD, CM

def Boeing_Vertol(
    CL,
    CD,
    CM,
    alpha,
    adotnorm,
    umach,
    Re,
    aoaStallPos,
    aoaStallNeg,
    AOA0,
    tc, 
    BV_DynamicFlagL,
    BV_DynamicFlagD,
    turbine,
    env,
    turbine_index,
    airfoil_index,
    family_factor=0.0,
):
    """
    Applies the Boeing–Vertol dynamic stall correction model.

    This function modifies the static aerodynamic coefficients using a
    Boeing–Vertol dynamic stall formulation. The static airfoil polar is
    assumed to be computed externally (e.g., via NeuralFoil) and provided
    as input coefficients (CL, CD, CM). When dynamic stall conditions are
    detected, corrected reference angles of attack are computed and the
    aerodynamic coefficients are updated accordingly.

    The model introduces a lag in the effective angle of attack based on
    the normalized angle-of-attack rate and Mach number effects.

    Parameters
    ----------
    CL : float
        Static lift coefficient at the current angle of attack.
    CD : float
        Static drag coefficient at the current angle of attack.
    CM : float
        Static moment coefficient at the current angle of attack.
    alpha : float
        Instantaneous angle of attack (radians).
    adotnorm : float
        Normalized angle-of-attack rate.
    umach : float
        Local Mach number.
    Re : float
        Reynolds number.
    aoaStallPos : float
        Positive stall angle (radians).
    aoaStallNeg : float
        Negative stall angle (radians).
    AOA0 : float
        Zero-lift angle of attack (radians).
    tc : float
        Airfoil thickness-to-chord ratio.
    BV_DynamicFlagL : int
        Lift dynamic stall flag (0 = off, 1 = active).
    BV_DynamicFlagD : int
        Drag dynamic stall flag (0 = off, 1 = active).
    turbine_index : int
        Turbine identifier used by the airfoil evaluation function.
    airfoil_index : int
        Airfoil identifier used by the airfoil evaluation function.
    family_factor : float, optional
        Airfoil family interpolation parameter, by default 0.0.

    Returns
    -------
    CL : float
        Lift coefficient after dynamic stall correction.
    CD : float
        Drag coefficient after dynamic stall correction.
    CM : float
        Moment coefficient after dynamic stall correction.
    BV_DynamicFlagL : int
        Updated lift dynamic stall flag.
    BV_DynamicFlagD : int
        Updated drag dynamic stall flag.

    Notes
    -----
    The static polar is expected to be computed outside this function.
    Calls to the airfoil evaluation function `af()` occur only when the
    dynamic stall model is active.

    The implementation follows a Boeing–Vertol-style dynamic stall model
    with Mach-dependent lag and stall-transition smoothing.
    """
    # Parameters
    k1pos = 0.5
    k1neg = 0.5
    diff = 0.06 - tc
    smachl = 0.4 + 5.0 * diff
    hmachl = 0.9 + 2.5 * diff
    gammaxl = 1.4 - 6.0 * diff
    dgammal = gammaxl / (hmachl - smachl)
    smachm = 0.2
    hmachm = 0.7 + 2.5 * diff
    gammaxm = 1.0 - 2.5 * diff
    dgammam = gammaxm / (hmachm - smachm)

    # Reference alpha limits
    Fac = 0.9
    dalphaRefMax = Fac * ksmin([abs_smooth(aoaStallPos - AOA0), abs_smooth(aoaStallNeg - AOA0)]) / ksmax([k1pos, k1neg])

    TransA = 0.5 * dalphaRefMax
    sign_adot = np.sign(adotnorm)

    # Lift model
    gammal = gammaxl - (umach - smachl) * dgammal
    dalphaLRef = gammal * np.sqrt(abs_smooth(adotnorm))
    dalphaLRef = ksmin([dalphaLRef, dalphaRefMax])

    if adotnorm * (alpha - AOA0) < 0.0:
        # CL magnitude decreasing
        dalphaL = k1neg * dalphaLRef
        alrefL = alpha - dalphaL * sign_adot

        if BV_DynamicFlagL == 1 and (aoaStallNeg < alrefL < aoaStallPos):
            BV_DynamicFlagL = 0
    else:
        # CL magnitude increasing
        dalphaL = k1pos * dalphaLRef
        alrefL = alpha - dalphaL * sign_adot

        if alpha <= aoaStallNeg or alpha >= aoaStallPos:
            BV_DynamicFlagL = 1
        else:
            BV_DynamicFlagL = 0

    # Drag model
    gammam = gammaxm - (umach - smachm) * dgammam
    if umach < smachm:
        gammam = gammaxm

    dalphaDRef = gammam * np.sqrt(abs_smooth(adotnorm))
    dalphaDRef = ksmin([dalphaDRef, dalphaRefMax])
    
    if adotnorm * (alpha - AOA0) < 0.0:
        dalphaD = k1neg * dalphaDRef
        alLagD = alpha - dalphaD * sign_adot

        if BV_DynamicFlagD == 1:
            delN = aoaStallNeg - alLagD
            delP = alLagD - aoaStallPos
        else:
            delN = delP = 0.0
    else:
        dalphaD = k1pos * dalphaDRef
        alLagD = alpha - dalphaD * sign_adot

        delN = aoaStallNeg - alpha
        delP = alpha - aoaStallPos

    if delN > TransA or delP > TransA:
        alrefD = alLagD
        BV_DynamicFlagD = 1
    elif 0 < delN < TransA:
        alrefD = alpha + (alLagD - alpha) * delN / TransA
        BV_DynamicFlagD = 1
    elif 0 < delP < TransA:
        alrefD = alpha + (alLagD - alpha) * delP / TransA
        BV_DynamicFlagD = 1
    else:
        BV_DynamicFlagD = 0

    # Dynamic stall corrections
    if BV_DynamicFlagL == 1:
        CL_ref, _, CM = af(alrefL, Re, umach, turbine, env, turbine_index, airfoil_index, family_factor)
        CL = CL_ref / (alrefL - AOA0) * (alpha - AOA0)

    if BV_DynamicFlagD == 1:
        _, CD, _ = af(alrefD, Re, umach, turbine, env, turbine_index, airfoil_index, family_factor)

    return CL, CD, CM, BV_DynamicFlagL, BV_DynamicFlagD



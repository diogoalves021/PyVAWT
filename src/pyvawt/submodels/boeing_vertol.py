import numpy as np

def abs_smooth(x, eps=1e-6):
    return np.sqrt(x*x + eps*eps)

def ksmin(values, k=100.0):
    values = np.array(values)
    return -np.log(np.sum(np.exp(-k * values))) / k

def ksmax(values, k=100.0):
    values = np.array(values)
    return np.log(np.sum(np.exp(k * values))) / k

def test_airfoil(alpha, Re, mach, family_factor=0.0):
    """
    Simplified symmetric airfoil model (for testing only)
    """
    Cl_alpha = 2 * np.pi        # rad^-1
    Cl = Cl_alpha * alpha

    Cd0 = 0.01
    Cd = Cd0 + 0.02 * alpha**2

    Cm = -0.02                 # constante (placeholder)

    return Cl, Cd, Cm

def Boeing_Vertol(
    af,
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
    family_factor=0.0
):

    # -------------------------
    # Parameters (plausible)
    # -------------------------
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

    # -------------------------
    # Reference alpha limits
    # -------------------------
    Fac = 0.9
    dalphaRefMax = (
        Fac
        * ksmin(
            [
                abs_smooth(aoaStallPos - AOA0),
                abs_smooth(aoaStallNeg - AOA0),
            ]
        )
        / ksmax([k1pos, k1neg])
    )

    TransA = 0.5 * dalphaRefMax
    sign_adot = np.sign(adotnorm) if adotnorm != 0 else 1.0

    # =========================================================
    # Lift model
    # =========================================================
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

    # =========================================================
    # Drag model
    # =========================================================
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

    # =========================================================
    # Static characteristics
    # =========================================================
    CL, CD, CM = af(alpha, Re, umach, family_factor)

    # =========================================================
    # Dynamic stall corrections
    # =========================================================
    if BV_DynamicFlagL == 1:
        CL_ref, _, CM = af(alrefL, Re, umach, family_factor)
        CL = CL_ref / (alrefL - AOA0) * (alpha - AOA0)

    if BV_DynamicFlagD == 1:
        _, CD, _ = af(alrefD, Re, umach, family_factor)

    return CL, CD, CM, BV_DynamicFlagL, BV_DynamicFlagD

class BoeingVertolAirfoilAdapter:
    """
    Adapter to make an Aerodynamics model compatible with the
    Boeing-Vertol dynamic stall interface.

    Expected callable signature:
        CL, CD, CM = af(alpha, Re, mach, family_factor)
    """

    def __init__(self, aero_model, cm0=-0.02):
        """
        Parameters
        ----------
        aero_model : Aerodynamics
            Any object implementing get_cl_cd(alpha, W)
        cm0 : float
            Constant (placeholder) pitching moment coefficient
        """
        self.aero = aero_model
        self.cm0 = cm0

    def __call__(self, alpha, Re=None, mach=None, family_factor=0.0):
        """
        Returns CL, CD, CM for Boeing-Vertol model.
        """
        # W is not used by FileAerodynamics; NeuralFoil may need it
        W = None

        CL, CD = self.aero.get_cl_cd(alpha, W)
        CM = self.cm0  # placeholder, physically plausible

        return CL, CD, CM

from ..simulation import NeuralFoilAerodynamics

aero = NeuralFoilAerodynamics(turbine_index=0, airfoil_index=0)

from boeingvertol import BoeingVertolAirfoilAdapter

af = BoeingVertolAirfoilAdapter(aero, cm0=-0.02)

CL, CD, CM, flagL, flagD = Boeing_Vertol(
    af=af,
    alpha=np.deg2rad(20),
    adotnorm=0.2,
    umach=0.15,
    Re=1e6,
    aoaStallPos=np.deg2rad(15),
    aoaStallNeg=np.deg2rad(-15),
    AOA0=0.0,
    tc=0.12,
    BV_DynamicFlagL=0,
    BV_DynamicFlagD=0
)



'''if __name__ == "__main__":

    alpha = np.deg2rad(45)
    adotnorm = 0.05
    umach = 0.1
    Re = 1e6

    aoaStallPos = np.deg2rad(15)
    aoaStallNeg = -np.deg2rad(15)
    AOA0 = 0.0
    tc = 0.12

    flagL = 0
    flagD = 0

    CL, CD, CM, flagL, flagD = Boeing_Vertol(
        test_airfoil,
        alpha,
        adotnorm,
        umach,
        Re,
        aoaStallPos,
        aoaStallNeg,
        AOA0,
        tc,
        flagL,
        flagD
    )

    print("CL =", CL)
    print("CD =", CD)
    print("CM =", CM)
    print("Flag L =", flagL)
    print("Flag D =", flagD)'''


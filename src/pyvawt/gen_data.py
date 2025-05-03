import numpy as np
import aerosandbox as asb

def get_single_cl_cd(airfoil, alpha, Reynolds, Mach, model_size='xxxlarge', include_360_deg_effects=False):
    """
    Returns CL and CD for a single angle of attack using NeuralFoil.

    Parameters
    ----------
    airfoil : asb.Airfoil
        The airfoil object from AeroSandbox.
    alpha : float
        Angle of attack in degrees.
    Reynolds : float
        Reynolds number.
    Mach : float
        Mach number.
    model_size : str, optional
        NeuralFoil model size. Defaults to 'xxxlarge'.
    include_360_deg_effects : bool, optional
        Whether to include 360° stall modeling. Defaults to False.

    Returns
    -------
    tuple of float
        CL and CD values.
    """
    af_data = airfoil.get_aero_from_neuralfoil(
        Re=Reynolds,
        mach=Mach,
        alpha=alpha,
        model_size=model_size,
        include_360_deg_effects=include_360_deg_effects
    )

    return af_data["CL"][0], af_data["CD"][0]


def get_multiple_cl_cd(airfoil, alpha, Reynolds, Mach, model_size='xxxlarge', include_360_deg_effects=False):
    """
    Returns arrays of CL and CD for multiple angles of attack using NeuralFoil.

    Parameters
    ----------
    airfoil : asb.Airfoil
        The airfoil object.
    alpha : array_like
        Angles of attack in degrees.
    Reynolds : array_like
        Reynolds numbers corresponding to each angle.
    Mach : array_like
        Mach numbers corresponding to each angle.
    model_size : str, optional
        NeuralFoil model size. Defaults to 'xxxlarge'.
    include_360_deg_effects : bool, optional
        Whether to include 360° stall modeling. Defaults to False.

    Returns
    -------
    tuple of ndarray
        Arrays of CL and CD values.
    """
    CL = np.zeros_like(alpha)
    CD = np.zeros_like(alpha)
    
    for i, (angle, Re, Ma) in enumerate(zip(alpha, Reynolds, Mach)):
        af_data = airfoil.get_aero_from_neuralfoil(
            Re=Re,
            mach=Ma,
            alpha=angle,
            model_size=model_size,
            include_360_deg_effects=include_360_deg_effects
        )
        
        CL[i] = af_data["CL"][0]
        CD[i] = af_data["CD"][0]

    return CL, CD

def readaerodyn_neuralfoil(airfoil_name="naca0012", Reynolds=1e6, Mach=0.0, model_size="xxxlarge", include_360_deg_effects=False):
    """
    Returns a function that provides CL and CD based on NeuralFoil predictions for a given alpha (in radians).

    Parameters
    ----------
    airfoil_name : str, optional
        Name of the airfoil as recognized by AeroSandbox.
    Reynolds : float, optional
        Reynolds number. Defaults to 1e6.
    Mach : float, optional
        Mach number. Defaults to 0.0.
    model_size : str, optional
        NeuralFoil model size. Defaults to 'xxxlarge'.
    include_360_deg_effects : bool, optional
        Whether to include 360° stall modeling. Defaults to False.

    Returns
    -------
    function
        A function `af(alpha_rad)` that returns CL and CD for one or more angles of attack (in radians).
    """
    airfoil = asb.Airfoil(name=airfoil_name)

    def af(alpha_rad):
        """
        Returns CL and CD for a given angle(s) of attack in radians.

        Parameters
        ----------
        alpha_rad : float or array_like
            Angle(s) of attack in radians.

        Returns
        -------
        tuple
            CL and CD values (float or array depending on input).
        """
        alpha_deg = np.degrees(alpha_rad)
        if np.isscalar(alpha_deg):
            cl, cd = get_single_cl_cd(
                airfoil=airfoil,
                alpha=alpha_deg,
                Reynolds=Reynolds,
                Mach=Mach,
                model_size=model_size,
                include_360_deg_effects=include_360_deg_effects
            )
        else:
            cl, cd = get_multiple_cl_cd(
                airfoil=airfoil,
                alpha=alpha_deg,
                Reynolds=np.full_like(alpha_deg, Reynolds),
                Mach=np.full_like(alpha_deg, Mach),
                model_size=model_size,
                include_360_deg_effects=include_360_deg_effects
            )
        return cl, cd

    return af


#af = readaerodyn('data/NACA_0012_mod.dat')
print('--' * 12)
print('\nInicializando a simulação...\n')
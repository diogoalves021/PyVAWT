import numpy as np
import aerosandbox as asb

def get_single_cl_cd(airfoil, alpha, Reynolds, Mach, model_size='xxxlarge', include_360_deg_effects=False):
    """
    Retorna CL e CD para um único valor de alpha (em graus), usando o NeuralFoil.
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
    Retorna arrays de CL e CD para múltiplos valores de alpha (em graus).
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
    Cria uma função af(alpha) baseada no modelo NeuralFoil em vez de ler um arquivo.
    """
    airfoil = asb.Airfoil(name=airfoil_name)

    def af(alpha_rad):
        """
        Retorna CL e CD para um alpha (ou array de alphas) em radianos.
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
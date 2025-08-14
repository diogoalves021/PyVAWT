import yaml
import numpy as np
import aerosandbox as asb


def load_config(path='src/pyvawt/config/config.yaml'):
    # Load YAML into a dict
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"O arquivo '{path}' está vazio ou inválido.")

    # Garante que todos os campos esperados existam
    for section in ['turbine', 'environment', 'simulation']:
        if section not in config:
            raise KeyError(f"Seção obrigatória '{section}' ausente no arquivo de configuração.")

    # Garante que certos campos sejam listas
    def ensure_list(section, key):
        if key in config[section] and not isinstance(config[section][key], list):
            config[section][key] = [config[section][key]]

    ensure_list('turbine', 'chord')
    ensure_list('turbine', 'solidity')
    ensure_list('environment', 'Vinf')
    ensure_list('simulation', 'airfoil')

    return config

def get_cl_cd_neuralfoil(alpha, W, turbine_index, airfoil_index):
    '''
    Return lift and drag coefficients using the NeuralFoil model.

    Parameters
    ----------
    alpha : float or array_like
        Angle of attack in radians. Can be a scalar or array; if array, `W`
        must have the same shape.
    W : float or array_like
        Relative wind speed at the section in m/s. Same shape as `alpha`.
    turbine_index : int
        Index of the turbine in the configuration (used to read chord).
    airfoil_index : int
        Index of the airfoil in the configuration (used to select the foil name).

    Returns
    -------
    cl : ndarray
        Lift coefficient(s) (C_L). NumPy array with the same shape as `alpha`.
    cd : ndarray
        Drag coefficient(s) (C_D). NumPy array with the same shape as `alpha`.

    Raises
    ------
    ValueError
        If `alpha` and `W` have different shapes.
    RuntimeError
        If the underlying NeuralFoil call (`asb.Airfoil.get_aero_from_neuralfoil`)
        fails or required configuration keys are missing.

    Notes
    -----
    - This function reads configuration using `load_config()` (so it depends on
      the presence and format of that configuration).
    - Local Reynolds number is computed as ``Re = rho * W * chord / mu`` and
      Mach number as ``mach = W / 343.2`` (speed of sound ~343.2 m/s).
    - `alpha` is converted from radians to degrees before calling the NeuralFoil
      API because the `get_aero_from_neuralfoil` method expects angles in degrees.
    - The function instantiates ``asb.Airfoil(name=...)`` on each call (no cache).
    - The NeuralFoil call uses ``model_size='xxxlarge'`` and reads
      ``include_360_deg_effects`` from the configuration.
    - Returned arrays are reshaped to match the input `alpha` shape.
    '''
    config = load_config()
    
    chord = config['turbine']['chord'][turbine_index]
    rho = config['environment']['rho']
    mu = config['environment']['mu']
    airfoil_name = config['simulation']['airfoil'][airfoil_index]

    Re = rho * W * chord / mu
    mach = W / 343.2

    airfoil = asb.Airfoil(name=airfoil_name)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=np.rad2deg(alpha),
        Re=Re,
        mach=mach,
        model_size=config['aero']['model_size'],
        include_360_deg_effects=config['simulation']['include_360_deg_effects']
    )

    cl = np.asarray(aero['CL']).reshape(alpha.shape)
    cd = np.asarray(aero['CD']).reshape(alpha.shape)
    return cl, cd


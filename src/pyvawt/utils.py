import yaml
import os
from .data_generation import get_cl_cd_neuralfoil

def load_config(path='src/pyvawt/config/config.yaml'):
    '''
    Loads the simulation configuration from a YAML file and returns it as a dictionary.

    Parameters
    ----------
    path : str, optional
        Path to the `.yaml` configuration file.
        Default is 'src/pyvawt/config/config.yaml'.

    Returns
    -------
    dict
        Dictionary containing the simulation parameters loaded from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML file.
    '''
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Configuration file not found: {path}')
    
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f'Error parsing YAML file {path}:\n{e}')

def save_config(config, path):
    '''
    Saves a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        Dictionary with simulation parameters to save.

    path : str
        Full path to the output `.yaml` file.
        If the file exists, it will be overwritten.

    Notes
    -----
    - Keys will be preserved in the original order (sort_keys=False).
    - Creates directories as needed.
    - Forces the file extension to .yaml if not present.
    - Uses indentation for human-readable output.
    '''
    if not path.endswith('.yaml'):
        path += '.yaml'

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

def get_cl_cd(alpha, W, turbine_index, airfoil_index, config):
    '''
    Retrieve lift (Cl) and drag (Cd) coefficients for a given angle of attack using
    the method specified in the configuration.

    Parameters
    ----------
    alpha : float
        Angle of attack [rad].
    W : float
        Relative wind speed at the section [m/s].
        Used only by the "neuralfoil" method.
    turbine_index : int
        Turbine index (used by the "neuralfoil" method).
    airfoil_index : int
        Airfoil index (used by the "neuralfoil" method).
    config : dict
        Simulation configuration dictionary.
        - ``config['aero']['method']`` : str
            Either "neuralfoil" (neural network model) or "file" (data interpolation).
        - ``config['aero']['af_func']`` : callable
            Function returning Cl and Cd given alpha (only used for the "file" method).

    Returns
    -------
    cl : float
        Lift coefficient.
    cd : float
        Drag coefficient.

    Raises
    ------
    ValueError
        If the aerodynamic coefficient method defined in ``config['aero']['method']`` is unknown.
    '''
    method = config['aero']['method']
    if method == 'neuralfoil':
        return get_cl_cd_neuralfoil(alpha, W, turbine_index, airfoil_index)
    elif method == 'file':
        af_func = config['aero']['af_func']
        return af_func(alpha)
    else:
        raise ValueError(f'Unknown Cl/Cd method: {method}')

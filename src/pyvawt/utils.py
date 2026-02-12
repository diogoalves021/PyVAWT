import yaml
import os
import numpy as np
import neuralfoil as nf
import aerosandbox as asb
from .data_generation import get_cl_cd_neuralfoil

def load_config(path):
    '''
    Loads the simulation configuration from a YAML file and returns it as a dictionary.

    Parameters
    ----------
    path : str, optional
        Path to the `.yaml` configuration file.

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

def read_dat(path):
    """
    Reads a .dat file and converts it into a list of lists of floats.
    Example: [[TSR, CP, CT, Rp, Tp, Zp], ...]
    """
    data = []
    with open(path, "r") as f:
        next(f)
        for line in f:
            if line.strip() == "":
                continue
            values = [float(x) for x in line.split()]
            data.append(values)
    return data

def mach(W):
    return(W / 343.2)

def get_tc_from_airfoil(config, airfoil_index):
    """
    Returns thickness-to-chord ratio (t/c) from airfoil defined in config.
    Works for 4 digit NACA airfoils.
    """

    airfoil_name = config['simulation']['airfoil'][airfoil_index].lower()

    if airfoil_name.startswith("naca") and len(airfoil_name) == 8:
        thickness_digits = airfoil_name[-2:]
        tc = int(thickness_digits) / 100.0
        return tc

    raise ValueError(f"Não foi possível determinar t/c para o aerofólio: {airfoil_name}")

def detect_stall_angles(config, airfoil_index):
    """
    Computes positive and negative stall angles using NeuralFoil.

    The airfoil is read from the config file.

    Returns
    -------
    aoaStallPos : float
        Positive stall angle [rad]
    aoaStallNeg : float
        Negative stall angle [rad]
    """

    # Airfoil from config
    airfoil_name = config['simulation']['airfoil'][airfoil_index].lower()
    airfoil = asb.Airfoil(airfoil_name)

    alpha_deg = np.linspace(-30, 30, 600)

    # Flow conditions
    Re = 2.5e6
    mach = 0.0
    model_size = "xxxlarge"

    # NeuralFoil
    aero = nf.get_aero_from_airfoil(
        airfoil=airfoil,
        alpha=alpha_deg,
        Re=Re,
        model_size=model_size
    )

    cl = aero["CL"]

    # Detect stall
    idx_pos = np.argmax(cl)
    idx_neg = np.argmin(cl)

    aoaStallPos = np.deg2rad(alpha_deg[idx_pos])
    aoaStallNeg = np.deg2rad(alpha_deg[idx_neg])

    return aoaStallPos, aoaStallNeg


import yaml
import os
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

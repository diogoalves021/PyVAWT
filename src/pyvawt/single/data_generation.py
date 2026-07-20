import yaml
import numpy as np
import aerosandbox as asb
from functools import lru_cache


@lru_cache(maxsize=1)
def load_config(path='src/pyvawt/config/config.yaml'):
    """Loads YAML into a dict, cached in memory to protect solver performance."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"O arquivo '{path}' está vazio ou inválido.")

    for section in ['turbine', 'environment', 'solver']:
        if section not in config:
            raise KeyError(f"Seção obrigatória '{section}' ausente no arquivo de configuração.")

    def ensure_list(section, key):
        if key in config[section] and not isinstance(config[section][key], list):
            config[section][key] = [config[section][key]]

    ensure_list('turbine', 'chord')
    ensure_list('turbine', 'solidity')
    ensure_list('environment', 'Vinf')
    
    if 'neuralfoil' in config['solver'] and 'airfoil' in config['solver']['neuralfoil']:
        nf_cfg = config['solver']['neuralfoil']
        if not isinstance(nf_cfg['airfoil'], list):
            nf_cfg['airfoil'] = [nf_cfg['airfoil']]

    return config


def get_cl_cd_neuralfoil(alpha, W, turbine_index, airfoil_index):
    """
    Return lift and drag coefficients using the NeuralFoil model.
    """
    config = load_config()
    
    chord = config['turbine']['chord'][turbine_index]
    rho = config['environment']['rho']
    mu = config['environment']['mu']
    
    airfoil_name = config['solver']['neuralfoil']['airfoil'][airfoil_index]

    Re = rho * W * chord / mu
    mach = W / 343.2

    airfoil = asb.Airfoil(name=airfoil_name)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=np.rad2deg(alpha),
        Re=Re,
        mach=mach,
        model_size=config['solver']['neuralfoil']['model_size'],
        include_360_deg_effects=False
    )

    cl = np.asarray(aero['CL']).reshape(alpha.shape)
    cd = np.asarray(aero['CD']).reshape(alpha.shape)
    return cl, cd

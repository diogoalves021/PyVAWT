from functools import lru_cache 
import numpy as np
import yaml
import aerosandbox as asb


@lru_cache(maxsize=1)
def load_config(path='src/pyvawt/config/config_multiple.yaml'):
    """Loads YAML into a dict, cached in memory to protect solver performance."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"O arquivo '{path}' está vazio ou inválido.")

    # Atualizado para validar as novas seções obrigatórias
    for section in ['turbine', 'environment', 'solver']:
        if section not in config:
            raise KeyError(f"Seção obrigatória '{section}' ausente no arquivo de configuração.")

    def ensure_list(target_dict, key):
        if key in target_dict and not isinstance(target_dict[key], list):
            target_dict[key] = [target_dict[key]]

    ensure_list(config['turbine'], 'chord')
    ensure_list(config['turbine'], 'solidity')
    ensure_list(config['environment'], 'Vinf')

    # Garante que 'airfoil' dentro de solver.neuralfoil seja tratado como lista
    nf_cfg = config.get('solver', {}).get('neuralfoil', {})
    ensure_list(nf_cfg, 'airfoil')

    return config


def get_cl_cd_neuralfoil(alpha, W, turbine_index, airfoil_index):
    """
    Return lift and drag coefficients using the NeuralFoil model based on the solver configuration.
    """
    config = load_config()
    
    # Extração resiliente de parâmetros da turbina e ambiente
    chords = config['turbine']['chord']
    chord = float(chords[turbine_index % len(chords)])
    
    rho = float(config['environment']['rho'])
    mu = float(config['environment']['mu'])

    # Acesso à subchave 'solver.neuralfoil'
    solver_cfg = config.get('solver', {})
    nf_cfg = solver_cfg.get('neuralfoil', {})

    # Leitura dos perfis e parâmetros específicos do NeuralFoil
    airfoils = nf_cfg.get('airfoil', ['naca0018'])
    airfoil_name = str(airfoils[airfoil_index % len(airfoils)])
    
    model_size = nf_cfg.get('model_size', 'large')
    include_360 = nf_cfg.get('include_360_deg_effects', True)

    Re = rho * W * chord / mu
    mach = W / 343.2

    airfoil = asb.Airfoil(name=airfoil_name)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=np.rad2deg(alpha),
        Re=Re,
        mach=mach,
        model_size=model_size,
        include_360_deg_effects=include_360
    )

    cl = np.asarray(aero['CL']).reshape(alpha.shape)
    cd = np.asarray(aero['CD']).reshape(alpha.shape)
    return cl, cd

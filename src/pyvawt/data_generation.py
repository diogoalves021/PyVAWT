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
    """
    Retorna os coeficientes de sustentação (cl) e arrasto (cd) usando o modelo NeuralFoil.

    Parâmetros:
    - alpha: ângulo de ataque [rad]
    - W: velocidade relativa [m/s]
    - config: dicionário com os parâmetros do JSON
    - turbine_index: índice da turbina no JSON
    - airfoil_index: índice do aerofólio no JSON

    Retorna:
    - cl, cd: coeficientes aerodinâmicos
    """
    config = load_config()
    

    chord = config["turbine"]["chord"][turbine_index]
    rho = config["environment"]["rho"]
    mu = config["environment"]["mu"]
    airfoil_name = config["simulation"]["airfoil"][airfoil_index]

    Re = rho * W * chord / mu
    mach = W / 343.2

    airfoil = asb.Airfoil(name=airfoil_name)
    aero = airfoil.get_aero_from_neuralfoil(
        alpha=np.rad2deg(alpha),
        Re=Re,
        mach=mach,
        model_size="xxxlarge",
        include_360_deg_effects=config["simulation"]["include_360_deg_effects"]
    )
    return aero["CL"], aero["CD"]


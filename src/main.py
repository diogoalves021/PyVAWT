import sys
import os
from read_data import readaerodyn
from simulation import Turbine, Environment, actuatorcylinder
from start_sim import load_config, initialize_turbine_and_environment, run_simulation

def main():
    # Carregar as configurações
    config = load_config()

    # Inicializar turbinas e ambiente
    turbines, env, simulation_params, turbine_params, environment_params, r, ntheta = initialize_turbine_and_environment(config)

    # Número de turbinas (pode ser 1 ou 2 conforme o config)
    num_turbines = simulation_params["num_turbines"]

    # Executar a simulação
    run_simulation(turbines, env, simulation_params, r, ntheta, environment_params["Vinf"], num_turbines, turbine_params)

if __name__ == "__main__":
    main()


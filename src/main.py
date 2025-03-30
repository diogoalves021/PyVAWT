'''
import sys
import os
from read_data import readaerodyn
from simulation import Turbine, Environment, actuatorcylinder
from start_sim import load_config, initialize_turbine_and_environment, run_simulation

def main():
    config = load_config()
    turbines, env, simulation_params, turbine_params, environment_params, r, ntheta = initialize_turbine_and_environment(config)
    num_turbines = simulation_params['num_turbines']
    run_simulation(turbines, env, simulation_params, r, ntheta, environment_params['Vinf'], num_turbines, turbine_params)

if __name__ == "__main__":
    main()

'''

import os
import sys

# Obter o diretório do arquivo atual (equivalente a splitdir(@__FILE__)[1] em Julia)
modulepath = os.path.dirname(__file__)

# Adiciona o diretório do módulo ao sys.path para que o Python possa encontrar os módulos
sys.path.append(modulepath)

# Importando os módulos
import read_data
import simulation
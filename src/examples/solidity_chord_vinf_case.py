import json
import time
import unittest
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from src.pyvawt import readaerodyn_neuralfoil, actuatorcylinder, Turbine, Environment

atol = 1e-6

def load_config(novo_perfil=None):
    """
    Loads the simulation configuration from a JSON file.

    Parameters
    ----------
    novo_perfil : str, optional
        Name of a new airfoil profile to override the one in the configuration file.

    Returns
    -------
    dict
        The loaded configuration dictionary, possibly modified with the new airfoil profile.
    """
    with open('src/pyvawt/config/config.json', 'r') as f:
        config = json.load(f)
    if novo_perfil:
        config['simulation']['aero_profile'] = novo_perfil
    return config

def save_config(config, path):
    """
    Saves the simulation configuration to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    path : str
        Destination path for the JSON file.
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def run_simulation_case(params):
    """
    Runs a single aerodynamic simulation case based on the provided parameters.

    Parameters
    ----------
    params : tuple
        A tuple containing the parameters:
        - airfoil : str
            Name of the airfoil profile.
        - chord : float
            Chord length of the blade (in meters).
        - solidity : float
            Solidity of the turbine.
        - vinf : float
            Freestream wind velocity (in m/s).

    Returns
    -------
    dict
        Dictionary summarizing the result of the simulation. Contains:
        - 'name' : str
            Folder name used to store the results.
        - 'airfoil', 'chord', 'solidity', 'vinf' : input parameters
        - 'status' : str
            'OK' if successful, or error message if failed.
        - 'time_sec' : float
            Duration of the simulation in seconds.

    Notes
    -----
    - The function initializes a turbine and environment, runs simulations across a TSR range,
      and stores results including a .dat file and a Cp vs TSR plot.
    - Assumes the use of 1 turbine for now.
    - Results are saved in a subdirectory of 'src/results/temporary_results'.
    """
    airfoil, chord, solidity, vinf = params
    B = 2
    r = chord * B / solidity
    start_time = time.time()

    # Nome da pasta baseado nos parâmetros
    folder_name = f'{airfoil}_ch{chord}_sol{solidity}_vinf{vinf}'.replace('.', 'p')
    result_dir = os.path.join('src/results/temporary_results', folder_name)
    os.makedirs(result_dir, exist_ok=True)

    try:
        print(f'Simulating: {folder_name}')

        # Carrega e configura
        config = load_config()
        config['simulation']['airfoil_name'] = airfoil
        config['turbine']['chord'] = chord
        config['turbine']['solidity'] = solidity
        config['environment']['Vinf'] = vinf
        config['turbine']['r'] = r

        save_config(config, os.path.join(result_dir, 'config_used.json'))

        # Inicializa simulação
        turbines, env, sim_params, turb_params, env_params, _, ntheta = initialize_turbine_and_environment(config)

        num_turbines = sim_params['num_turbines']
        var_omega_vinf = sim_params['var_omega_vinf']
        profile_name = airfoil

        # =====
        # Simulação para uma turbina
        # =====

        if num_turbines==1:
            n = 20
            tsrvec = np.linspace(1, 7, n)
            CPvec = np.zeros(n)
            CTvec = np.zeros(n)
            Rpvec = np.zeros(n)
            Tpvec = np.zeros(n)
            Zpvec = np.zeros(n)
            thetavec = np.zeros((n, ntheta))

            if var_omega_vinf == 0:
                for i, tsr in enumerate(tsrvec):
                    turbines[0].Omega = vinf * tsr / r
                    CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)
                    CPvec[i] = CP[0]
                    CTvec[i] = CT[0]
                    Rpvec[i] = Rp[0].item()
                    Tpvec[i] = Tp[0].item()
                    Zpvec[i] = Zp[0].item()
                    thetavec[i, :] = theta

            elif var_omega_vinf == 1:
                for i, tsr in enumerate(tsrvec):
                    turbines[0].Omega = 13.62 * 2 * np.pi / 60.0
                    env.Vinf = turbines[0].Omega * r / tsr
                    CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)
                    CPvec[i] = CP[0]
                    CTvec[i] = CT[0]
                    Rpvec[i] = Rp[0].item()
                    Tpvec[i] = Tp[0].item()
                    Zpvec[i] = Zp[0].item()
                    thetavec[i, :] = theta
            
            else:
                print(f'[ERROR] var_omega_vinf invalid: {var_omega_vinf}')
                return
            
            # Salvar resutados como .dat dentro da pasta
            data_to_save = np.column_stack((tsrvec, CPvec, CTvec, Rpvec, Tpvec, Zpvec))
            header = 'TSR\tCP\tCT\tRp\tTp\tZp'
            out_filename = os.path.join(result_dir, f'results_{profile_name}.dat')
            np.savetxt(out_filename, data_to_save, header=header, fmt='%.6f', delimiter='\t')

            # Salvar gráfico como imagem
            plt.figure(figsize=(10, 5))
            plt.plot(tsrvec, CPvec, color='blue', label='$C_p$')
            plt.title(f'$C_p$ x TSR para {profile_name}')
            plt.xlabel('TSR')
            plt.ylabel('$C_p$')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plot_filename = os.path.join(result_dir, f'cp_curve_{profile_name}.eps')
            plt.savefig(plot_filename)
            plt.close()

        elapsed = time.time() - start_time
        return {
            'name': folder_name,
            'airfoil': airfoil,
            'chord': chord,
            'solidity': solidity,
            'vinf': vinf,
            'status': 'OK',
            'time_sec': round(elapsed, 2)
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'name': folder_name,
            'airfoil': airfoil,
            'chord': chord,
            'solidity': solidity,
            'vinf': vinf,
            'status': f'ERROR: {e}',
            'time_sec': round(elapsed, 2)
        }

def initialize_turbine_and_environment(config):
    """
    Initializes the turbine and environment objects based on the configuration file.

    Parameters
    ----------
    config : dict
        Dictionary containing simulation, turbine, and environment parameters.

    Returns
    -------
    turbines : list of Turbine
        List containing the initialized Turbine object(s).
    env : Environment
        The Environment object initialized with freestream conditions.
    simulation_params : dict
        Dictionary with general simulation parameters.
    turbine_params : dict
        Dictionary with turbine-specific parameters.
    environment_params : dict
        Dictionary with environmental parameters.
    r : float
        Rotor radius (in meters).
    ntheta : int
        Number of azimuthal discretization points.

    Notes
    -----
    The function also reads airfoil data using the `readaerodyn_neuralfoil` function
    and uses it to initialize the turbine's aerodynamic properties.
    """
    turbine_params = config['turbine']
    environment_params = config['environment']
    simulation_params = config['simulation']

    # Parâmetros da turbina
    r = turbine_params['r']
    twist = turbine_params['twist']
    delta = turbine_params['delta']
    af = af = readaerodyn_neuralfoil(
    airfoil_name=simulation_params.get('airfoil_name', 'naca0021'), 
    Reynolds=simulation_params.get('Re', 5e6),
    Mach=simulation_params.get('Mach', 0.3),
    model_size=simulation_params.get('model_size', 'xxxlarge'),
    include_360_deg_effects=simulation_params.get('include_360_deg_effects', False)
)
    chord = turbine_params['chord']
    B = turbine_params['B']
    solidity = turbine_params['solidity']
    centerX = turbine_params['centerX']
    centerY = turbine_params['centerY']
    Omega = turbine_params['Omega']
    ntheta = turbine_params['ntheta']

    # Parâmetros do ambiente
    Vinf = environment_params['Vinf']
    rho = environment_params['rho']
    mu = environment_params['mu']

    # Criação da turbina
    turbines = [None] * 1
    turbines[0] = Turbine(r, chord, twist, delta, B, af, Omega, 0.0, 0.0)

    # Criação do ambiente
    env = Environment(Vinf, rho, mu)

    return turbines, env, simulation_params, turbine_params, environment_params, r, ntheta

def runtest():
    """
    Runs a batch of aerodynamic simulations for different parameter combinations.

    The function defines a small set of test cases using airfoil, chord, solidity,
    and freestream velocity values, and runs them in parallel using multiple CPU cores.

    Results are saved to disk:
    - Individual simulation results are stored in subfolders under 'src/results/temporary_results'.
    - A log file named 'log_simulacoes.csv' is saved summarizing all simulations.

    Notes
    -----
    - Uses ProcessPoolExecutor for parallel execution of simulations.
    - The list of parameters can be modified directly within the function.
    - This is the main function to be run as a script.
    """
    # Lista dos parâmetros a serem testados
    airfoils = ['naca0018']
    #chords = [0.875, 1.3125, 1.75, 2.1875, 2.625] #simulação para corda de 0.875 e suas combinações feitas.
    #chords = [1.3125, 1.75, 2.1875, 2.625]
    #solidities = [0.1, 0.3, 0.5, 0.7]
    #vinfs = [3.115, 4.6725, 6.23, 7.7875, 9.345]
    chords = [1.3125]
    solidities = [0.1]
    vinfs = [9.345]


    combinations = list(product(airfoils, chords, solidities, vinfs))
    print(f'Initializing {len(combinations)} paralel simulations... \n')

    log_data = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(run_simulation_case, p): p for p in combinations}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Simulações'):
            result = future.result()
            log_data.append(result)
            print(result['name'], '-', result['status'])

    # Salvar log em csv
    log_path = os.path.join('src/results/temporary_results', 'log_simulacoes.csv')
    with open(log_path, mode='w', newline='') as csvfile:
        fieldnames = ['name', 'airfoil', 'chord', 'solidity', 'vinf', 'status', 'time_sec']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_data)

    print(f'\n Log salvo em: {log_path}')

    # Garante que a pasta de resultados exista
    os.makedirs("src/results/temporary_results", exist_ok=True)

if __name__ == "__main__":
    runtest()

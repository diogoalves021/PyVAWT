import json
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.pyvawt import actuatorcylinder, Turbine, Environment


def load_config(path="src/pyvawt/config/config.json"):
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
    with open(path, "r") as f:
        return json.load(f)


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
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def initialize_turbine_and_environment(config):
    """
    Initializes the turbine and environment objects based on the configuration file.

    Parameters
    ----------
    config : dict
        Dictionary containing simulation, turbine, and environment parameters.

    Returns
    -------
    turbine : Turbine
        The initialized Turbine object.
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
    turbine_params = config["turbine"]
    environment_params = config["environment"]
    simulation_params = config["simulation"]

    r = turbine_params["r"]
    twist = turbine_params["twist"]
    delta = turbine_params["delta"]
    chord = turbine_params["chord"]
    B = turbine_params["B"]
    solidity = turbine_params["solidity"]
    centerX = turbine_params["centerX"]
    centerY = turbine_params["centerY"]
    Omega = turbine_params["Omega"]
    ntheta = turbine_params["ntheta"]

    Vinf = environment_params["Vinf"]
    rho = environment_params["rho"]
    mu = environment_params["mu"]

    turbine = Turbine(r, chord, twist, delta, B, Omega, centerX, centerY)
    env = Environment(Vinf, rho, mu)

    return turbine, env, simulation_params, turbine_params, environment_params, r, ntheta


def run_simulation_case(params, base_config):
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
    airfoil_index, turbine_index, chord, solidity, vinf = params
    config = json.loads(json.dumps(base_config))  # deep copy

    airfoil_name = config["simulation"]["airfoil"][airfoil_index]
    config["simulation"]["airfoil"] = airfoil_name
    config["turbine"]["chord"] = chord
    config["turbine"]["solidity"] = solidity
    config["environment"]["Vinf"] = vinf
    angular_velocity = config["turbine"]["Omega"]

    folder_name = (
        f"{airfoil_name}_ch{chord}_sol{solidity}_vinf{vinf}".replace(".", "p")
    )
    result_dir = os.path.join("src/results/temporary_results", folder_name)
    os.makedirs(result_dir, exist_ok=True)
    save_config(config, os.path.join(result_dir, "config_used.json"))

    turbine, env, sim_params, _, _, _, ntheta = initialize_turbine_and_environment(
        config
    )
    r = config["turbine"]["r"]
    var_omega_vinf = sim_params["var_omega_vinf"]
    num_turbines = sim_params["num_turbines"]

    start_time = time.time()
    try:
        print(f"Simulando: {folder_name}")
        n = 20
        tsrvec = np.linspace(1, 7, n)
        CPvec = np.zeros(n)
        CTvec = np.zeros(n)
        Rpvec = np.zeros(n)
        Tpvec = np.zeros(n)
        Zpvec = np.zeros(n)

        if var_omega_vinf == 0:
            for i, tsr in enumerate(tsrvec):
                turbine.Omega = vinf * tsr / r
                CT, CP, Rp, Tp, Zp, _ = actuatorcylinder(
                    turbine, env, ntheta, config, turbine_index, airfoil_index
                )
                CPvec[i], CTvec[i], Rpvec[i], Tpvec[i], Zpvec[i] = (
                    CP,
                    CT,
                    Rp[0],
                    Tp[0],
                    Zp[0],
                )

        elif var_omega_vinf == 1:
            for i, tsr in enumerate(tsrvec):
                turbine.Omega = angular_velocity
                env.Vinf = turbine.Omega * r / tsr
                CT, CP, Rp, Tp, Zp, _ = actuatorcylinder(
                    turbine, env, ntheta, config, turbine_index, airfoil_index
                )
                CPvec[i], CTvec[i], Rpvec[i], Tpvec[i], Zpvec[i] = (
                    CP,
                    CT,
                    Rp[0],
                    Tp[0],
                    Zp[0],
                )

        else:
            raise ValueError("var_omega_vinf inválido.")

        data_to_save = np.column_stack((tsrvec, CPvec, CTvec, Rpvec, Tpvec, Zpvec))
        header = "TSR\tCP\tCT\tRp\tTp\tZp"
        np.savetxt(
            os.path.join(result_dir, f"results_{airfoil_name}.dat"),
            data_to_save,
            header=header,
            fmt="%.6f",
            delimiter="\t",
        )

        plt.figure()
        plt.plot(tsrvec, CPvec)
        plt.title(f"$C_p$ x TSR - {airfoil_name}")
        plt.xlabel("TSR")
        plt.ylabel("$C_p$")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"cp_curve_{airfoil_name}.png"))
        plt.close()

        elapsed = time.time() - start_time
        return {"name": folder_name, "status": "OK", "time_sec": round(elapsed, 2)}

    except Exception as e:
        return {
            "name": folder_name,
            "status": f"ERROR: {e}",
            "time_sec": round(time.time() - start_time, 2),
        }


def run_simulation():
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
    config = load_config()

    airfoil_indices = list(range(len(config["simulation"]["airfoil"])))
    turbine_indices = list(range(config["simulation"]["num_turbines"]))
    chords = config["turbine"]["chord"]
    solidities = config["turbine"]["solidity"]
    vinfs = config["environment"]["Vinf"]

    combinations = []
    for ai in airfoil_indices:
        for ti in turbine_indices:
            for chord in chords:
                for sol in solidities:
                    for vinf in vinfs:
                        combinations.append((ai, ti, chord, sol, vinf))

    print(f"Iniciando {len(combinations)} simulações paralelas...\n")

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(run_simulation_case, params, config): params
            for params in combinations
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulações"):
            results.append(future.result())

    log_path = os.path.join("src/results/temporary_results", "log_simulacoes.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLog salvo em: {log_path}")


if __name__ == "__main__":
    run_simulation()


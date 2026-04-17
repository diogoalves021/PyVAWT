import yaml
import os
import numpy as np
import neuralfoil as nf
import aerosandbox as asb
import argparse
import time
from tabulate import tabulate

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

def get_tc_from_airfoil(airfoil_name: str):
    """
    Returns thickness-to-chord ratio (t/c) from airfoil name.
    Works for 4 digit NACA airfoils.
    """
    airfoil_name = airfoil_name.lower()

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
    mach = 3.0
    model_size = "large"

    # NeuralFoil
    aero = nf.get_aero_from_airfoil(
        airfoil=airfoil,
        alpha=alpha_deg,
        Re=Re,
        model_size=model_size
    )

    cl = aero["CL"]

    # Detect airfoil_name stall
    idx_pos = np.argmax(cl)
    idx_neg = np.argmin(cl)

    aoaStallPos = np.deg2rad(alpha_deg[idx_pos])
    aoaStallNeg = np.deg2rad(alpha_deg[idx_neg])

    aoaStallPos_deg = np.degrees(aoaStallPos)

    if not np.isclose(aoaStallPos, -aoaStallNeg, rtol=1e-9):
        raise ValueError("Stall angles error: stall angles are not symmetric.")

    print(f'{airfoil_name.upper()} stall angle: ± {aoaStallPos_deg:.1f}°')
    return aoaStallPos, aoaStallNeg

def print_config(config):
    print("\nSimulation parameters")
    print("=" * 40)

    for section, params in config.items():
        print(f"\n{section.upper()}")
        print("-" * 40)

        for key, value in params.items():
            print(f"{key:<20} : {value}")

def print_summary(config):
    print("\nSimulation summary")
    print("=" * 40)

    turb = config.get("turbine", {})
    env = config.get("environment", {})
    sim = config.get("simulation", {})
    aero = config.get("aero", {})

    print("\nTURBINE")
    print("-" * 40)
    print(f"radius (r)           : {turb.get('r')}")
    print(f"height (H)           : {turb.get('H')}")
    print(f"blades (B)           : {turb.get('B')}")
    print(f"chord                : {turb.get('chord')}")
    print(f"theta discretization : {turb.get('ntheta')}")

    print("\nENVIRONMENT")
    print("-" * 40)
    print(f"wind speed (Vinf)    : {env.get('Vinf')}")
    print(f"density (rho)        : {env.get('rho')}")

    print("\nSIMULATION")
    print("-" * 40)
    print(f"airfoil              : {sim.get('airfoil')}")
    print(f"Reynolds number      : {sim.get('Re')}")
    print(f"Mach number          : {sim.get('Mach')}")
    print(f"dynamic stall        : {aero.get('dynamic_stall')}")

    print("=" * 40)

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run VAWT actuator-cylinder simulations."
    )

    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to YAML configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show full configuration before running simulation"
    )

    return parser.parse_args()

def print_simulation_footer(results, start_time, log_path):
    """
    Print a clear final summary of the simulation run.

    Parameters
    ----------
    results : list[dict]
        List containing the result dictionaries from each simulation case.
    start_time : float
        Timestamp from the beginning of the simulation (time.time()).
    log_path : str
        Path to the CSV file where results were saved.
    """

    total_cases = len(results)
    successful = sum(1 for r in results if r.get("status") == "OK")
    failed = total_cases - successful

    total_time = time.time() - start_time
    mins = int(total_time // 60)
    secs = int(total_time % 60)

    avg_time = total_time / total_cases if total_cases else 0
    avg_mins = int(avg_time // 60)
    avg_secs = int(avg_time % 60)

    print("\n" + "=" * 60)
    print("Simulation finished")
    print("=" * 60)

    print(f"Cases executed : {total_cases}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {failed}")

    print("\nTiming")
    print("-" * 60)
    print(f"Total runtime  : {mins:02d}:{secs:02d}")
    print(f"Average case   : {avg_mins:02d}:{avg_secs:02d}")

    print("\nResults saved to")
    print("-" * 60)
    print(log_path)

    print("=" * 60 + "\n")

def format_time(seconds):
    try:
        seconds = float(seconds)
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{minutes:02d}:{secs:02d}'
    except Exception:
        return str(seconds)

def print_simulation_results(results, start_time, log_path):
    """
    Print the final simulation results table and summary statistics.
    """

    print()
    print("=" * 40)
    print("SIMULATION RESULTS")
    print("=" * 40)

    if not results:
        print("No results to show.")
        print(f"Log file : {log_path}")
        print("=" * 40)
        return

    headers = ["Case", "Status", "Time (MM:SS)"]
    rows = []

    for r in results:
        time_val = format_time(r.get("time_sec", ""))
        rows.append((r.get("name", ""), r.get("status", ""), time_val))

    print()
    print(tabulate(rows, headers=headers, tablefmt="plain"))

    # ---- statistics ----
    total_cases = len(results)
    successful = sum(1 for r in results if r.get("status") == "OK")
    failed = total_cases - successful

    total_time = time.time() - start_time
    mins = int(total_time // 60)
    secs = int(total_time % 60)

    avg_time = total_time / total_cases if total_cases else 0
    avg_mins = int(avg_time // 60)
    avg_secs = int(avg_time % 60)

    print("-" * 40)
    print(f"Cases executed : {total_cases}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {failed}")
    print()
    print(f"Total runtime  : {mins:02d}:{secs:02d}")
    print(f"Average case   : {avg_mins:02d}:{avg_secs:02d}")
    print()
    print(f"Log file       : {log_path}")
    print("=" * 40)

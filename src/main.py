"""
Aerodynamic Simulation Runner for Coupled VAWT Turbines.
This module orchestrates parameter sweeps, manages file I/O, and generates 
scientific-quality reports of simulation runs.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from src.simulation import actuatorcylinder, Turbine, Environment
from src.read_data import readaerodyn

# Global constants
ATOL: float = 1e-6

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class SimulationContext(NamedTuple):
    """Structured container for initialized simulation physical components."""
    turbines: List[Turbine]
    env: Environment
    simulation_params: Dict[str, Any]
    turbine_params: Dict[str, Any]
    environment_params: Dict[str, Any]
    radius: float
    ntheta: int


def load_config(path: Union[str, Path] = 'config/config.json') -> Dict[str, Any]:
    """Loads the simulation configuration from a JSON file.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        json.JSONDecodeError: If the configuration file format is invalid.
    """
    config_path = Path(path)
    try:
        with config_path.open('r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except json.JSONDecodeError as err:
        logger.error(f"Invalid JSON format in {config_path}: {err}")
        raise


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Saves the simulation configuration back to a JSON file."""
    config_path = Path(path)
    try:
        with config_path.open('w', encoding='utf-8') as file:
            json.dump(config, file, indent=4)
    except IOError as err:
        logger.error(f"Failed to write configuration to {config_path}: {err}")
        raise


def _apply_plot_style() -> None:
    """Applies high-quality serif typography styles for academic/technical plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
    })


def plot_turbine_layout(turbines: List[Turbine], results_dir: Path, case_name: str) -> None:
    """Generates a high-resolution top-view plot of the spatial layout of the turbines."""
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i, t in enumerate(turbines):
        # Coordinates of physical turbine center (X, Y)
        ax.plot(t.centerX, t.centerY, 'xr', markersize=10, label='Center' if i == 0 else "")
        # DASHED outer perimeter of rotational diameter
        circle = plt.Circle((t.centerX, t.centerY), t.r, color='blue', fill=False, linestyle='--', alpha=0.6)
        ax.add_patch(circle)
        # Visual ID Tag
        ax.text(t.centerX, t.centerY + t.r * 1.1, f"Turbine {i+1}", ha='center', fontsize=12, weight='bold')
        
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    
    # Save results utilizing robust Pathlib
    png_path = results_dir / f'layout_{case_name}.png'
    pdf_path = results_dir / f'layout_{case_name}.pdf'
    
    fig.savefig(png_path, dpi=700)
    fig.savefig(pdf_path)
    plt.close(fig)


def _execute_tsr_sweep(
    turbines: List[Turbine],
    env: Environment,
    ntheta: int,
    tsr_vec: np.ndarray,
    var_omega_vinf: int,
    vinf: float,
    radius: float,
    num_turbines: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs the sequential sweep over all TSR values implementing the physical Warm Start solver."""
    n_points = len(tsr_vec)
    cp_vec = np.zeros((n_points, num_turbines))
    ct_vec = np.zeros((n_points, num_turbines))
    rp_vec = np.zeros((n_points, num_turbines))
    tp_vec = np.zeros((n_points, num_turbines))
    zp_vec = np.zeros((n_points, num_turbines))
    theta_vec = np.zeros((n_points, ntheta))

    # Warm start pointer initialization
    w_guess = None

    for i, tsr in enumerate(tsr_vec):
        if var_omega_vinf == 0:
            for turbine in turbines:
                turbine.Omega = vinf * tsr / radius
        elif var_omega_vinf == 1:
            for turbine in turbines:
                turbine.Omega = 13.62 * 2 * np.pi / 60.0
            env.Vinf = turbines[0].Omega * radius / tsr
        else:
            raise ValueError(f"Invalid var_omega_vinf strategy configured: {var_omega_vinf}")
        
        # Actuator Cylinder evaluation with previous solution seeding (Warm Start)
        ct, cp, rp, tp, zp, theta, w_guess = actuatorcylinder(turbines, env, ntheta, w0=w_guess)
        
        # Mapping results for the current iteration step
        for t in range(num_turbines):
            cp_vec[i, t] = cp[t]
            ct_vec[i, t] = ct[t]
            rp_vec[i, t] = rp[0, t]
            tp_vec[i, t] = tp[0, t]
            zp_vec[i, t] = zp[0, t]
        theta_vec[i, :] = theta

    return cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec


def _save_raw_data(
    results_dir: Path,
    case_name: str,
    num_turbines: int,
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray,
    ct_vec: np.ndarray,
    rp_vec: np.ndarray,
    tp_vec: np.ndarray,
    zp_vec: np.ndarray
) -> None:
    """Saves raw data results (.dat files) sequentially for each modeled turbine."""
    for t in range(num_turbines):
        data_to_save = np.column_stack((
            tsr_vec, 
            cp_vec[:, t], 
            ct_vec[:, t], 
            rp_vec[:, t], 
            tp_vec[:, t], 
            zp_vec[:, t]
        ))
        header = 'TSR\tCP\tCT\tRp\tTp\tZp'
        out_filename = results_dir / f'results_{case_name}_t{t+1}.dat'
        np.savetxt(out_filename, data_to_save, header=header, fmt='%.6f', delimiter='\t')


def _plot_performance_curves(
    results_dir: Path,
    case_name: str,
    num_turbines: int,
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray
) -> None:
    """Generates clean publication-ready performance curves (Cp vs. TSR)."""
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot individual curves (filter-out negative values)
    for t in range(num_turbines):
        cp_t = cp_vec[:, t]
        mask = cp_t >= 0
        idx_sort = np.argsort(tsr_vec[mask])
        ax.plot(
            tsr_vec[mask][idx_sort], 
            cp_t[mask][idx_sort], 
            marker='o', 
            label=f'Turbine {t+1}'
        )
    
    # Calculate and overlay system-average if running multiple turbines
    if num_turbines > 1:
        avg_cp = np.mean(cp_vec, axis=1)
        mask_avg = avg_cp >= 0
        idx_sort_avg = np.argsort(tsr_vec[mask_avg])
        ax.plot(
            tsr_vec[mask_avg][idx_sort_avg], 
            avg_cp[mask_avg][idx_sort_avg], 
            '--', 
            color='black', 
            linewidth=2, 
            label='System Average'
        )

    ax.set_xlabel("TSR ($\lambda$)")
    ax.set_ylabel(r"$C_p$")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    
    png_path = results_dir / f'cp_curve_{case_name}.png'
    pdf_path = results_dir / f'cp_curve_{case_name}.pdf'
    
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def run_simulation_case(params: Tuple[int, int, float, float, float]) -> Dict[str, Any]:
    """Runs an aerodynamic simulation case supporting coupled turbines with Warm Start.

    Args:
        params: Config parameters unpackable to (airfoil_idx, turbine_idx, chord, solidity, vinf)

    Returns:
        Dict: Metadata report on final execution statistics.
    """
    _, _, chord, solidity, vinf = params
    config = load_config()
    
    # Airfoil path parsing
    airfoil_path = Path(config['simulation']['aero_profile'])
    airfoil_name = airfoil_path.stem
    
    # Dynamic parameter updating based on execution sweep
    config['simulation']['aero_profile'] = str(airfoil_path)
    config['turbine']['chord'] = chord
    config['turbine']['solidity'] = solidity
    config['environment']['Vinf'] = vinf
    config['turbine']['r'] = chord * config['turbine']['B'] / solidity

    # Case file-system identifier string
    case_name = f'{airfoil_name}_ch{chord}_sol{solidity}_vinf{vinf}'.replace('.', 'p')

    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Export execution config setup for traceablity
    save_config(config, results_dir / f'config_{case_name}.json')

    # Initialize domain physical variables
    context = initialize_turbine_and_environment(config)
    
    plot_turbine_layout(context.turbines, results_dir, case_name)

    num_turbines = int(context.simulation_params['num_turbines'])
    var_omega_vinf = int(context.simulation_params['var_omega_vinf'])

    start_time = time.perf_counter()

    try:
        logger.info(f"Simulating {num_turbines} turbine(s) using case: {case_name}")

        # Build TSR Sweep Range
        n_points = 20
        tsr_vec = np.linspace(1.0, 7.0, n_points)
        
        # Compute calculations via AC-Method
        cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec = _execute_tsr_sweep(
            turbines=context.turbines,
            env=context.env,
            ntheta=context.ntheta,
            tsr_vec=tsr_vec,
            var_omega_vinf=var_omega_vinf,
            vinf=vinf,
            radius=context.radius,
            num_turbines=num_turbines
        )

        # Write output raw files
        _save_raw_data(
            results_dir=results_dir,
            case_name=case_name,
            num_turbines=num_turbines,
            tsr_vec=tsr_vec,
            cp_vec=cp_vec,
            ct_vec=ct_vec,
            rp_vec=rp_vec,
            tp_vec=tp_vec,
            zp_vec=zp_vec
        )

        # Draw plots
        _plot_performance_curves(
            results_dir=results_dir,
            case_name=case_name,
            num_turbines=num_turbines,
            tsr_vec=tsr_vec,
            cp_vec=cp_vec
        )

        elapsed = time.perf_counter() - start_time
        logger.info(f"Finished. Output files successfully written to '{results_dir}/'")

        return {
            'name': case_name,
            'airfoil': airfoil_name,
            'chord': chord,
            'solidity': solidity,
            'vinf': vinf,
            'status': 'OK',
            'time_sec': round(elapsed, 2)
        }
        
    except Exception as err:
        logger.error(f"Execution crashed for case {case_name}. Traceback attached:", exc_info=True)
        raise


def initialize_turbine_and_environment(config: Dict[str, Any]) -> SimulationContext:
    """Initializes the turbine list and environment objects based on the configuration file."""
    turbine_params = config["turbine"]
    environment_params = config["environment"]
    simulation_params = config["simulation"]

    r = float(turbine_params["r"])
    twist = float(turbine_params["twist"])
    delta = float(turbine_params["delta"])
    chord = float(turbine_params["chord"])
    B = int(turbine_params["B"])
    centerX = turbine_params["centerX"]
    centerY = turbine_params["centerY"]
    Omega = float(turbine_params["Omega"])
    ntheta = int(turbine_params["ntheta"])

    Vinf = float(environment_params["Vinf"])
    rho = float(environment_params["rho"])
    mu = float(environment_params["mu"])

    # Core aerodynamic function load
    af = readaerodyn(simulation_params["aero_profile"])
    num_turbines = int(simulation_params.get("num_turbines", 1))

    # Automatic grid spacing if centering arrays are defined dynamically as single floats
    if isinstance(centerX, (int, float)):
        cx_list = [float(centerX) + i * 4.0 * r for i in range(num_turbines)]
    else:
        cx_list = [float(x) for x in centerX]

    if isinstance(centerY, (int, float)):
        cy_list = [float(centerY) for _ in range(num_turbines)]
    else:
        cy_list = [float(y) for y in centerY]

    # Pad physical layout lists if user configurations are partially initialized
    if len(cx_list) < num_turbines:
        cx_list.extend([cx_list[-1] + 4.0 * r for _ in range(num_turbines - len(cx_list))])
    if len(cy_list) < num_turbines:
        cy_list.extend([cy_list[-1] for _ in range(num_turbines - len(cy_list))])

    # Build Turbines instances
    turbines = [
        Turbine(r, chord, twist, delta, B, af, Omega, cx_list[i], cy_list[i])
        for i in range(num_turbines)
    ]

    # Instantiate Environment
    env = Environment(Vinf, rho, mu)

    return SimulationContext(
        turbines=turbines,
        env=env,
        simulation_params=simulation_params,
        turbine_params=turbine_params,
        environment_params=environment_params,
        radius=r,
        ntheta=ntheta
    )


def main() -> None:
    """Main execution thread."""
    logger.info("Initializing system simulation setup...\n")
    try:
        config = load_config()

        chord = float(config["turbine"]["chord"])
        solidity = float(config["turbine"]["solidity"])
        vinf = float(config["environment"]["Vinf"])

        # Call simulation case run
        result = run_simulation_case((0, 0, chord, solidity, vinf))

        print("\n" + "=" * 45)
        print(f"{'SIMULATION PIPELINE SUMMARY':^45}")
        print("=" * 45)
        for key, value in result.items():
            print(f"{key:<15}: {value}")
        print("=" * 45 + "\n")

    except Exception as err:
        logger.critical(f"Pipeline crashed. Uncaught fatal error: {err}", exc_info=True)


if __name__ == "__main__":
    main()

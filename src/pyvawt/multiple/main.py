"""
Aerodynamic Simulation Runner for Coupled VAWT Turbines.
Delegates NeuralFoil evaluations to the modular 'data_generation' library.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, NamedTuple
import yaml

import matplotlib.pyplot as plt
import numpy as np

from src.pyvawt.multiple.simulation import actuatorcylinder, Turbine, Environment
from src.pyvawt.multiple.read_data import readaerodyn

from src.pyvawt.multiple.data_generation import load_config, get_cl_cd_neuralfoil

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


class NeuralFoilAirfoilWrapper:
    """
    An adapter that interfaces the classic solver with your 'data_generation.py' module.
    It translates calls expecting (alpha, Re) into your get_cl_cd_neuralfoil(alpha, W) signature.
    """
    def __init__(self, turbine_index: int, airfoil_index: int):
        self.turbine_index = turbine_index
        self.airfoil_index = airfoil_index

    def get_coefficients(self, alpha_rad: Union[float, np.ndarray], Re: Union[float, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        config = load_config()
        
        # CORRIGIDO: Tratamento dinâmico e seguro para evitar IndexError na leitura da corda
        chord_entry = config['turbine']['chord']
        if isinstance(chord_entry, list):
            chord = float(chord_entry[self.turbine_index % len(chord_entry)])
        else:
            chord = float(chord_entry)
            
        rho = config['environment']['rho']
        mu = config['environment']['mu']

        # Se o solver não fornecer Re, assume uma velocidade relativa padrão
        if Re is None:
            W = 10.0
        else:
            Re = np.asarray(Re)
            # Re = rho * W * chord / mu  =>  W = Re * mu / (rho * chord)
            W = Re * mu / (rho * chord)

        # Delega o cálculo para o seu módulo 'data_generation.py'
        return get_cl_cd_neuralfoil(alpha_rad, W, self.turbine_index, self.airfoil_index)

    def __call__(self, alpha_rad, Re=None):
        return self.get_coefficients(alpha_rad, Re)


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
        ax.plot(t.centerX, t.centerY, 'xr', markersize=10, label='Center' if i == 0 else "")
        circle = plt.Circle((t.centerX, t.centerY), t.r, color='blue', fill=False, linestyle='--', alpha=0.6)
        ax.add_patch(circle)
        ax.text(t.centerX, t.centerY + t.r * 1.1, f"Turbine {i+1}", ha='center', fontsize=12, weight='bold')
        
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    
    fig.savefig(results_dir / f'layout_{case_name}.png', dpi=700)
    fig.savefig(results_dir / f'layout_{case_name}.pdf')
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
        
        ct, cp, rp, tp, zp, theta, w_guess = actuatorcylinder(turbines, env, ntheta, w0=w_guess)
        
        for t in range(num_turbines):
            cp_vec[i, t] = cp[t]
            ct_vec[i, t] = ct[t]
            
            # CORRIGIDO: Extração dimensional robusta para rp, tp, zp (previne quebras de formato)
            if rp.ndim == 2:
                rp_vec[i, t] = rp[0, t]
                tp_vec[i, t] = tp[0, t]
                zp_vec[i, t] = zp[0, t]
            else:
                rp_vec[i, t] = rp[t]
                tp_vec[i, t] = tp[t]
                zp_vec[i, t] = zp[t]
                
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

    for t in range(num_turbines):
        cp_t = cp_vec[:, t]
        mask = cp_t >= -1.0
        idx_sort = np.argsort(tsr_vec[mask])
        ax.plot(
            tsr_vec[mask][idx_sort], 
            cp_t[mask][idx_sort], 
            marker='o', 
            label=f'Turbine {t+1}'
        )
    
    if num_turbines > 1:
        avg_cp = np.mean(cp_vec, axis=1)
        mask_avg = avg_cp >= -1.0
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
    
    fig.savefig(results_dir / f'cp_curve_{case_name}.png', dpi=300)
    fig.savefig(results_dir / f'cp_curve_{case_name}.pdf')
    plt.close(fig)


def run_simulation_case(params: Tuple[int, int, float, float, float]) -> Dict[str, Any]:
    """Runs an aerodynamic simulation case supporting coupled turbines with NeuralFoil integration."""
    _, _, chord, solidity, vinf = params
    config = load_config()
    
    # Extração a partir do novo bloco 'solver'
    solver_cfg = config.get("solver", {})
    method = solver_cfg.get("method", "neuralfoil")
    use_neuralfoil = (method == "neuralfoil")
    
    if use_neuralfoil:
        nf_cfg = solver_cfg.get("neuralfoil", {})
        raw_airfoil = nf_cfg.get("airfoil", ["naca0018"])
        airfoil_name = raw_airfoil[0] if isinstance(raw_airfoil, list) else str(raw_airfoil)
    else:
        file_cfg = solver_cfg.get("file", {})
        airfoil_file = file_cfg.get("path")
        if not airfoil_file:
            raise KeyError("Caminho do perfil clássico não encontrado em 'solver.file.path'")
        airfoil_path = Path(airfoil_file)
        airfoil_name = airfoil_path.stem

    num_turbines = int(solver_cfg.get("num_turbines", 1))
    
    # Mapeamento do parâmetro fixo ('vinf' -> estratégia 0 | 'omega' -> estratégia 1)
    fixed_param = str(solver_cfg.get("fixed_parameter", "vinf")).lower()
    var_omega_vinf = 0 if fixed_param == "vinf" else 1

    config['turbine']['chord'] = [chord] * num_turbines
    config['turbine']['solidity'] = [solidity] * num_turbines
    config['environment']['Vinf'] = [vinf] if isinstance(config['environment']['Vinf'], list) else vinf
    config['turbine']['r'] = chord * config['turbine']['B'] / solidity

    case_name = f'{airfoil_name}_ch{chord}_sol{solidity}_vinf{vinf}'.replace('.', 'p')

    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f'config_{case_name}.yaml', 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    context = initialize_turbine_and_environment(config)
    plot_turbine_layout(context.turbines, results_dir, case_name)

    start_time = time.perf_counter()

    try:
        logger.info(f"Simulating {num_turbines} turbine(s) using case: {case_name}")

        n_points = 20
        tsr_vec = np.linspace(1.0, 7.0, n_points)
        
        cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec = _execute_tsr_sweep(
            turbines=context.turbines,
            env=context.env,
            ntheta=context.ntheta,
            tsr_vec=tsr_vec,
            var_omega_vinf=var_omega_vinf,  # Usa o mapeamento de fixed_parameter
            vinf=vinf,
            radius=context.radius,
            num_turbines=num_turbines
        )

        _save_raw_data(results_dir, case_name, num_turbines, tsr_vec, cp_vec, ct_vec, rp_vec, tp_vec, zp_vec)
        _plot_performance_curves(results_dir, case_name, num_turbines, tsr_vec, cp_vec)

        elapsed = time.perf_counter() - start_time
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
        logger.error(f"Execution crashed for case {case_name}.", exc_info=True)
        raise

def initialize_turbine_and_environment(config: Dict[str, Any]) -> SimulationContext:
    """Initializes the turbine list and environment objects using solver YAML configuration."""
    turbine_params = config["turbine"]
    environment_params = config["environment"]
    solver_params = config.get("solver", {})

    r = float(turbine_params["r"])
    twist = float(turbine_params["twist"])
    delta = float(turbine_params["delta"])
    B = int(turbine_params["B"])
    centerX = turbine_params["centerX"]
    centerY = turbine_params["centerY"]
    Omega = float(turbine_params["Omega"])
    ntheta = int(turbine_params["ntheta"])

    Vinf = float(environment_params["Vinf"][0]) if isinstance(environment_params["Vinf"], list) else float(environment_params["Vinf"])
    rho = float(environment_params["rho"])
    mu = float(environment_params["mu"])

    num_turbines = int(solver_params.get("num_turbines", 1))

    raw_chord = turbine_params["chord"]
    if isinstance(raw_chord, list):
        chord_list = [float(c) for c in raw_chord]
    else:
        chord_list = [float(raw_chord)] * num_turbines
    if len(chord_list) < num_turbines:
        chord_list.extend([chord_list[-1]] * (num_turbines - len(chord_list)))

    # Leitura baseada no novo bloco solver:
    method = solver_params.get("method", "neuralfoil")
    use_neuralfoil = (method == "neuralfoil")
    turbines_airfoils = []

    if use_neuralfoil:
        for i in range(num_turbines):
            wrapper = NeuralFoilAirfoilWrapper(turbine_index=i, airfoil_index=i)
            turbines_airfoils.append(wrapper)
    else:
        file_cfg = solver_params.get("file", {})
        aero_profile = file_cfg.get("path")
        if not aero_profile:
            raise KeyError("Caminho do arquivo não encontrado em 'solver.file.path'")
            
        classic_profile = readaerodyn(aero_profile)
        turbines_airfoils = [classic_profile] * num_turbines

    if isinstance(centerX, (int, float)):
        cx_list = [float(centerX) + i * 4.0 * r for i in range(num_turbines)]
    else:
        cx_list = [float(x) for x in centerX]

    if isinstance(centerY, (int, float)):
        cy_list = [float(centerY) for _ in range(num_turbines)]
    else:
        cy_list = [float(y) for y in centerY]

    if len(cx_list) < num_turbines:
        cx_list.extend([cx_list[-1] + 4.0 * r for _ in range(num_turbines - len(cx_list))])
    if len(cy_list) < num_turbines:
        cy_list.extend([cy_list[-1] for _ in range(num_turbines - len(cy_list))])

    turbines = [
        Turbine(r, chord_list[i], twist, delta, B, turbines_airfoils[i], Omega, cx_list[i], cy_list[i])
        for i in range(num_turbines)
    ]

    env = Environment(Vinf, rho, mu)

    return SimulationContext(
        turbines=turbines,
        env=env,
        simulation_params=solver_params,
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

        chord_val = config["turbine"]["chord"][0] if isinstance(config["turbine"]["chord"], list) else float(config["turbine"]["chord"])
        solidity_val = config["turbine"]["solidity"][0] if isinstance(config["turbine"]["solidity"], list) else float(config["turbine"]["solidity"])
        vinf_val = config["environment"]["Vinf"][0] if isinstance(config["environment"]["Vinf"], list) else float(config["environment"]["Vinf"])

        result = run_simulation_case((0, 0, chord_val, solidity_val, vinf_val))

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

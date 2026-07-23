"""
Aerodynamic Simulation Runner for Coupled VAWT Turbines.
Delegates NeuralFoil evaluations to the modular 'data_generation' library.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, NamedTuple, Optional
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
        
        chord_entry = config['turbine']['chord']
        if isinstance(chord_entry, list):
            chord = float(chord_entry[self.turbine_index % len(chord_entry)])
        else:
            chord = float(chord_entry)
            
        rho = config['environment']['rho']
        mu = config['environment']['mu']

        if Re is None:
            W = 10.0
        else:
            Re = np.asarray(Re)
            W = Re * mu / (rho * chord)

        return get_cl_cd_neuralfoil(alpha_rad, W, self.turbine_index, self.airfoil_index)

    def __call__(self, alpha_rad, Re=None):
        return self.get_coefficients(alpha_rad, Re)


# ==============================================================================
# CAMADA DE I/O E EXPORTAÇÃO DE RESULTADOS (EXPORTER COM SUPORTE A CONFIG)
# ==============================================================================

def _apply_plot_style() -> None:
    """Applies high-quality serif typography styles for academic/technical plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
    })


def plot_turbine_layout(turbines: List[Turbine], case_dir: Path, fmt: str = 'png', dpi: int = 300) -> None:
    """Generates a top-view plot of the spatial layout using user-defined format and DPI."""
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
    
    out_path = case_dir / f'layout.{fmt.lower()}'
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_raw_data(
    case_dir: Path,
    num_turbines: int,
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray,
    ct_vec: np.ndarray,
    rp_vec: np.ndarray,
    tp_vec: np.ndarray,
    zp_vec: np.ndarray,
    fmt: str = 'dat',
    include_header: bool = True
) -> None:
    """Saves raw data results (.dat or .csv) with optional headers and customizable delimiters."""
    is_csv = (fmt.lower() == 'csv')
    delimiter = ',' if is_csv else '\t'
    ext = 'csv' if is_csv else 'dat'

    if include_header:
        header = f'TSR{delimiter}CP{delimiter}CT{delimiter}Rp{delimiter}Tp{delimiter}Zp'
    else:
        header = ''

    for t in range(num_turbines):
        data_to_save = np.column_stack((
            tsr_vec, 
            cp_vec[:, t], 
            ct_vec[:, t], 
            rp_vec[:, t], 
            tp_vec[:, t], 
            zp_vec[:, t]
        ))
        out_filename = case_dir / f'results_t{t+1}.{ext}'
        np.savetxt(out_filename, data_to_save, header=header, fmt='%.6f', delimiter=delimiter, comments='')


def _plot_performance_curves(
    case_dir: Path,
    num_turbines: int,
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray,
    fmt: str = 'png',
    dpi: int = 300
) -> None:
    """Generates performance curves (Cp vs. TSR) using user-defined format and DPI."""
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
    
    out_path = case_dir / f'cp_curve.{fmt.lower()}'
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def export_coupled_case_results(
    case_name: str,
    config: Dict[str, Any],
    turbines: List[Turbine],
    tsr_vec: np.ndarray,
    cp_vec: np.ndarray,
    ct_vec: np.ndarray,
    rp_vec: np.ndarray,
    tp_vec: np.ndarray,
    zp_vec: np.ndarray,
    base_results_dir: Union[str, Path] = 'results'
) -> Optional[Path]:
    """
    Exporta os resultados respeitando integralmente o bloco 'output' do config.yaml.
    """
    output_cfg = config.get("output", {})

    # 1. Verifica se o salvamento está habilitado
    if not output_cfg.get("save", True):
        logger.info("[IO] Salvamento de resultados desativado (output.save = false).")
        return None

    # Extração das opções de salvamento com fallbacks seguros
    save_config_flag = output_cfg.get("save_config", True)
    save_plot_flag = output_cfg.get("save_plot", True)

    data_file_cfg = output_cfg.get("data_file", {})
    data_fmt = data_file_cfg.get("format", "dat")
    inc_header = data_file_cfg.get("include_header", True)

    plot_img_cfg = output_cfg.get("plot_image", {})
    img_fmt = plot_img_cfg.get("format", "png")
    img_dpi = int(plot_img_cfg.get("dpi", 300))

    # Criação do diretório do caso
    case_dir = Path(base_results_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # 2. Salva a cópia do YAML de configuração
    if save_config_flag:
        with open(case_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    # 3. Exporta arquivos brutos de dados (.dat ou .csv)
    num_turbines = len(turbines)
    _save_raw_data(
        case_dir=case_dir,
        num_turbines=num_turbines,
        tsr_vec=tsr_vec,
        cp_vec=cp_vec,
        ct_vec=ct_vec,
        rp_vec=rp_vec,
        tp_vec=tp_vec,
        zp_vec=zp_vec,
        fmt=data_fmt,
        include_header=inc_header
    )

    # 4. Gera os gráficos caso a flag save_plot esteja ativa
    if save_plot_flag:
        plot_turbine_layout(turbines, case_dir, fmt=img_fmt, dpi=img_dpi)
        _plot_performance_curves(case_dir, num_turbines, tsr_vec, cp_vec, fmt=img_fmt, dpi=img_dpi)

    logger.info(f"[IO] Resultados salvos com sucesso na pasta: {case_dir.resolve()}")
    return case_dir


# ==============================================================================
# CAMADA DE SOLVER E FÍSICA
# ==============================================================================

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


def run_simulation_case(params: Tuple[int, int, float, float, float]) -> Dict[str, Any]:
    """Runs an aerodynamic simulation case supporting coupled turbines with NeuralFoil integration."""
    _, _, chord, solidity, vinf = params
    config = load_config()
    
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
    blades = int(config.get("turbine", {}).get("B", 3))
    
    # Cálculo do Raio
    radius = round(chord * blades / solidity, 4)

    fixed_param = str(solver_cfg.get("fixed_parameter", "vinf")).lower()
    var_omega_vinf = 0 if fixed_param == "vinf" else 1

    config['turbine']['chord'] = [chord] * num_turbines
    config['turbine']['solidity'] = [solidity] * num_turbines
    config['environment']['Vinf'] = [vinf] if isinstance(config['environment']['Vinf'], list) else vinf
    config['turbine']['r'] = radius

    case_name = f'{airfoil_name}_turb{num_turbines}_b{blades}_r{radius}_ch{chord}_sol{solidity}_vinf{vinf}'.replace('.', 'p')

    context = initialize_turbine_and_environment(config)
    start_time = time.perf_counter()

    try:
        logger.info(f"Simulating {num_turbines} turbine(s) [B={blades}, R={radius}m] using case: {case_name}")

        n_points = 20
        tsr_vec = np.linspace(1.0, 7.0, n_points)
        
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

        elapsed = time.perf_counter() - start_time

        # Exporta tudo para a subpasta do caso respeitando as opções do config.yaml
        export_coupled_case_results(
            case_name=case_name,
            config=config,
            turbines=context.turbines,
            tsr_vec=tsr_vec,
            cp_vec=cp_vec,
            ct_vec=ct_vec,
            rp_vec=rp_vec,
            tp_vec=tp_vec,
            zp_vec=zp_vec,
            base_results_dir='results'
        )

        return {
            'name': case_name,
            'airfoil': airfoil_name,
            'num_turbines': num_turbines,
            'blades': blades,
            'radius': radius,
            'chord': chord,
            'solidity': solidity,
            'vinf': vinf,
            'status': 'OK',
            'time_sec': round(elapsed, 2)
        }
        
    except Exception as err:
        logger.error(f"Execution crashed for case {case_name}.", exc_info=True)
        raise


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

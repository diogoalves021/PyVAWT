"""
Aerodynamic Simulation Runner for Coupled VAWT Turbines.
Delegates NeuralFoil evaluations to the modular 'data_generation' library.
"""
import os
import logging
import time
import copy
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, NamedTuple, Optional
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pyvawt.multiple.simulation import actuatorcylinder, Turbine, Environment
from src.pyvawt.multiple.read_data import readaerodyn
from src.pyvawt.multiple.data_generation import load_config, get_cl_cd_neuralfoil

# Global constants
ATOL: float = 1e-6

# Configuração básica do logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# ==============================================================================
# ESTRUTURAS DE DADOS E ADAPTERS AERODINÂMICOS
# ==============================================================================

class SimulationContext(NamedTuple):
    """Structured container for initialized simulation physical components."""
    turbines: List[Turbine]
    env: Environment
    simulation_params: Dict[str, Any]
    turbine_params: Dict[str, Any]
    environment_params: Dict[str, Any]
    radius: float
    ntheta: int


from numba import njit

# ==============================================================================
# KERNEL NUMBA DE ALTÍSSIMA VELOCIDADE (C-LEVEL INTERPOLATION)
# ==============================================================================

@njit(fastmath=True, cache=True)
def _bilinear_interp_2d_numba(
    alpha_wrapped: np.ndarray,
    w_clamped: np.ndarray,
    values: np.ndarray,
    alpha_min: float,
    inv_dalpha: float,
    log_w_min: float,
    inv_dlog_w: float,
    n_alpha: int,
    n_w: int
) -> np.ndarray:
    """
    Kernel otimizado em nível de C compilado via Numba.
    Executa a interpolação bilinear sem alocações temporárias na RAM.
    """
    alpha_flat = alpha_wrapped.ravel()
    w_flat = w_clamped.ravel()
    n = alpha_flat.size
    out = np.empty(n, dtype=np.float64)

    max_i = n_alpha - 1.000001
    max_j = n_w - 1.000001

    for k in range(n):
        # Mapeamento O(1) direto de índices
        fi = (alpha_flat[k] - alpha_min) * inv_dalpha
        fj = (np.log(w_flat[k]) - log_w_min) * inv_dlog_w

        # Bounds Clamping
        if fi < 0.0:
            fi = 0.0
        elif fi > max_i:
            fi = max_i

        if fj < 0.0:
            fj = 0.0
        elif fj > max_j:
            fj = max_j

        i0 = int(fi)
        j0 = int(fj)
        i1 = i0 + 1
        j1 = j0 + 1

        t = fi - i0
        u = fj - j0

        v00 = values[i0, j0]
        v10 = values[i1, j0]
        v01 = values[i0, j1]
        v11 = values[i1, j1]

        out[k] = (1.0 - t) * (1.0 - u) * v00 + t * (1.0 - u) * v10 + (1.0 - t) * u * v01 + t * u * v11

    return out.reshape(alpha_wrapped.shape)


class FastBilinear2D:
    """
    Wrapper Python para o Kernel compilado em Numba.
    """
    def __init__(self, alpha_grid: np.ndarray, w_grid: np.ndarray, values: np.ndarray):
        self.n_alpha = len(alpha_grid)
        self.n_w = len(w_grid)
        self.values = np.ascontiguousarray(values, dtype=np.float64)

        self.alpha_min = float(alpha_grid[0])
        self.inv_dalpha = float((self.n_alpha - 1) / (alpha_grid[-1] - alpha_grid[0]))

        self.log_w_min = float(np.log(w_grid[0]))
        self.inv_dlog_w = float((self.n_w - 1) / (np.log(w_grid[-1]) - np.log(w_grid[0])))

    def __call__(self, alpha_wrapped: np.ndarray, w_clamped: np.ndarray) -> np.ndarray:
        return _bilinear_interp_2d_numba(
            alpha_wrapped,
            w_clamped,
            self.values,
            self.alpha_min,
            self.inv_dalpha,
            self.log_w_min,
            self.inv_dlog_w,
            self.n_alpha,
            self.n_w
        )

class NeuralFoilAirfoilWrapper:
    """
    Adapter otimizado com Look-Up Table 2D, Cache em Disco e Interpolação O(1).
    """
    def __init__(self, turbine_index: int, airfoil_index: int, n_alpha: int = 1800, n_w: int = 40):
        self.turbine_index = turbine_index
        self.airfoil_index = airfoil_index
        
        config = load_config()
        
        chord_entry = config['turbine']['chord']
        if isinstance(chord_entry, list):
            self.chord = float(chord_entry[self.turbine_index % len(chord_entry)])
        else:
            self.chord = float(chord_entry)
            
        self.rho = float(config['environment']['rho'])
        self.mu = float(config['environment']['mu'])

        # Sistema de Cache em Disco
        cache_dir = Path(".cache_lut")
        cache_dir.mkdir(exist_ok=True)
        
        cache_sig = f"t{turbine_index}_a{airfoil_index}_c{self.chord:.6f}_r{self.rho:.4f}_m{self.mu:.4e}_na{n_alpha}_nw{n_w}"
        hash_id = hashlib.md5(cache_sig.encode('utf-8')).hexdigest()[:10]
        cache_file = cache_dir / f"lut_{hash_id}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file)
                self.alpha_grid = data['alpha_grid']
                self.w_grid = data['w_grid']
                self.cl_grid = data['cl_grid']
                self.cd_grid = data['cd_grid']
            except Exception:
                self._build_and_cache_lut(n_alpha, n_w, cache_file)
        else:
            self._build_and_cache_lut(n_alpha, n_w, cache_file)

        # Interpoladores customizados de altíssima velocidade
        self._interp_cl = FastBilinear2D(self.alpha_grid, self.w_grid, self.cl_grid)
        self._interp_cd = FastBilinear2D(self.alpha_grid, self.w_grid, self.cd_grid)

    def _build_and_cache_lut(self, n_alpha: int, n_w: int, cache_file: Path) -> None:
        self.alpha_grid = np.linspace(-np.pi, np.pi, n_alpha)
        self.w_grid = np.geomspace(0.1, 150.0, n_w)

        ALPHA, W = np.meshgrid(self.alpha_grid, self.w_grid, indexing='ij')
        
        cl_flat, cd_flat = get_cl_cd_neuralfoil(
            ALPHA.ravel(), 
            W.ravel(), 
            self.turbine_index, 
            self.airfoil_index
        )

        self.cl_grid = np.asarray(cl_flat, dtype=np.float64).reshape(ALPHA.shape)
        self.cd_grid = np.asarray(cd_flat, dtype=np.float64).reshape(ALPHA.shape)

        np.savez_compressed(
            cache_file, 
            alpha_grid=self.alpha_grid, 
            w_grid=self.w_grid, 
            cl_grid=self.cl_grid, 
            cd_grid=self.cd_grid
        )

    def get_coefficients(
        self, 
        alpha_rad: Union[float, np.ndarray], 
        Re: Union[float, np.ndarray] = None
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        alpha_arr = np.asarray(alpha_rad, dtype=np.float64)
        alpha_wrapped = (alpha_arr + np.pi) % (2.0 * np.pi) - np.pi

        if Re is None:
            w_arr = np.full_like(alpha_wrapped, 10.0)
        else:
            w_arr = np.asarray(Re, dtype=np.float64) * (self.mu / (self.rho * self.chord))

        w_clamped = np.clip(w_arr, self.w_grid[0], self.w_grid[-1])

        # Consulta ultrarrápida O(1)
        cl = self._interp_cl(alpha_wrapped, w_clamped)
        cd = self._interp_cd(alpha_wrapped, w_clamped)

        if alpha_arr.ndim == 0:
            return float(cl), float(cd)

        return cl, cd

    def __call__(self, alpha_rad, Re=None):
        return self.get_coefficients(alpha_rad, Re)

# ==============================================================================
# CAMADA DE I/O E EXPORTAÇÃO DE RESULTADOS
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

    header = f'TSR{delimiter}CP{delimiter}CT{delimiter}Rp{delimiter}Tp{delimiter}Zp' if include_header else ''

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
    """Exporta os resultados respeitando integralmente o bloco 'output' do config.yaml."""
    output_cfg = config.get("output", {})

    if not output_cfg.get("save", True):
        logger.info("[IO] Salvamento de resultados desativado (output.save = false).")
        return None

    save_config_flag = output_cfg.get("save_config", True)
    save_plot_flag = output_cfg.get("save_plot", True)

    data_file_cfg = output_cfg.get("data_file", {})
    data_fmt = data_file_cfg.get("format", "dat")
    inc_header = data_file_cfg.get("include_header", True)

    plot_img_cfg = output_cfg.get("plot_image", {})
    img_fmt = plot_img_cfg.get("format", "png")
    img_dpi = int(plot_img_cfg.get("dpi", 300))

    case_dir = Path(base_results_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    if save_config_flag:
        with open(case_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

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
    """
    Executa a varredura sequencial do sweep de TSR.
    Preserva o Warm Start (w0 = w_guess) reduzindo as iterações do solver de ~18 para ~3 por ponto.
    """
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
            raise ValueError(f"Estratégia var_omega_vinf inválida configurada: {var_omega_vinf}")
        
        # Warm Start mantido: reutiliza a matriz de velocidades induzidas do ponto anterior
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

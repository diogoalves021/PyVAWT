"""Main entry point for PYVAWT aerodynamic simulations."""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pyvawt.single.aerodynamics import detect_stall_angles
from src.pyvawt.single.export import (
    export_2d_results,
    export_3d_results,
    BASE_RESULTS_DIR
)
from src.pyvawt.single.simulation import (
    run_simulation_case,
    simulate_3D_turbine,
)
from src.pyvawt.single.utils import (
    _worker_init,
    load_config,
    parse_args,
    warmup_numba_kernels,
)
from src.pyvawt.ui.ui import (
    UI,
    format_time,
    print_simulation_config,
    print_simulation_footer,
    print_simulation_results,
    print_summary,
)

# Execution command example:
# uv run python3 -m src.pyvawt.single.main


def run_simulation(config_path: str = "src/pyvawt/config/config.yaml") -> None:
    """
    Execute a batch of aerodynamic simulations specified by a configuration file.

    Handles pre-processing initialization, stall angle detection, and routing
    for either 3D multi-slice evaluations or 2D parallel parameter sweeps.

    Parameters
    ----------
    config_path : str, default="src/pyvawt/config/config.yaml"
        Fallback file path to the default simulation configuration YAML file if
        no CLI argument is provided.

    Returns
    -------
    None

    See Also
    --------
    run_simulation_case : Solves a single 2D simulation instance.
    simulate_3D_turbine : Executes vertical discretization for 3D mode.
    """
    # 1. Header & Initialization
    UI.banner("PYVAWT - ONE-TURBINE SOLVER MODULE")

    UI.section("INITIALIZATION & PRE-PROCESSING")
    warmup_numba_kernels(verbose=True)

    args = parse_args()
    config = load_config(path=args.config or config_path)

    # Compute static stall angles for all configured airfoils
    stall_angles = {}
    for airfoil_index in range(len(config["solver"]["neuralfoil"]["airfoil"])):
        aoaStallPos, aoaStallNeg = detect_stall_angles(config, airfoil_index)
        stall_angles[airfoil_index] = (aoaStallPos, aoaStallNeg)

    # Optional configuration printout
    if args.show_config:
        print_simulation_config(config)

    # =========================================================================
    # 2. CHECK 3D SIMULATION MODE
    # =========================================================================
    sim3d_cfg = config.get("solver", {}).get("simulation3d", {})

    if sim3d_cfg.get("enabled", False):
        res_3d = simulate_3D_turbine(config, stall_angles)

        export_3d_results(
            tsr=res_3d["tsr"],
            cp_3d=res_3d["cp_3d"],
            config=config,
            output_dir=res_3d["result_dir"],
        )
        return

    # =========================================================================
    # 3. 2D PARALLEL SWEEP MODE
    # =========================================================================
    UI.section("EXECUTION (2D SWEEP)")

    airfoil_indices = list(range(len(config["solver"]["neuralfoil"]["airfoil"])))
    chords = config["turbine"]["chord"]
    solidities = config["turbine"]["solidity"]
    vinfs = config["environment"]["Vinf"]
    flow_cfg = config.get("submodels", {}).get("flow_curvature", {})

    combinations = [
        (ai, 0, chord, sol, vinf)
        for ai in airfoil_indices
        for chord in chords
        for sol in solidities
        for vinf in vinfs
    ]

    total = len(combinations)
    results = []
    start_time = time.perf_counter()

    PARALLEL_THRESHOLD = 33  # Limiar para evitar overhead do ProcessPoolExecutor em poucos casos

    # --- Single-case execution ---
    if total == 1:
        UI.status("Simulation Cases", "1 combination (Single-Core Direct Execution)")
        params = combinations[0]
        
        result = run_simulation_case(
            params, 
            config, 
            flow_cfg, 
            stall_angles, 
            output_dir=None,  # Usa o padrão YYYYMMDD_HHMMSS_2D_solX.X
            show_details=True
        )
        results.append(result)

    # --- Multi-case parametric sweep (Cria pasta pai + subpastas organizadas) ---
    else:
        # Diretório raiz para a varredura paramétrica (100% idêntico ao original)
        sweep_timestamp = time.strftime("%Y%m%d_%H%M%S")
        sweep_base_dir = BASE_RESULTS_DIR / f"{sweep_timestamp}_2D_sweep"

        # --- Execução Sequencial Rápida (Para 2 ou 3 casos: sem overhead de multiprocessing) ---
        if total < PARALLEL_THRESHOLD:
            UI.status("Simulation Cases", f"{total} combinations (Fast Sequential Execution)")
            print()

            for idx, params in enumerate(combinations):
                ai, _, chord, sol, vinf = params
                airfoil_name = config["solver"]["neuralfoil"]["airfoil"][ai]
                
                # Subpasta individual mantida exatamente igual
                case_dir = sweep_base_dir / f"{airfoil_name}_c{chord:.2f}_sol{sol:.3f}_v{vinf:.2f}"

                try:
                    result = run_simulation_case(
                        params, 
                        config, 
                        flow_cfg, 
                        stall_angles, 
                        output_dir=case_dir,  # Mesma estrutura de salvamento
                        show_details=False
                    )
                except Exception as e:
                    result = {
                        "name": str(params),
                        "status": "ERROR",
                        "time_sec": 0.0,
                        "error": repr(e),
                    }

                results.append(result)
                now = time.perf_counter()
                UI.progress_bar(idx + 1, total, now - start_time, prefix="2D Sweep")

        # --- Execução Paralela (Para 4 ou mais casos) ---
        else:
            num_workers = min(total, os.cpu_count() or 1)
            UI.status("Simulation Cases", f"{total} combinations")
            UI.status("Active CPU Cores", f"{num_workers}")
            print()

            completed = 0
            last_update = 0.0

            with ProcessPoolExecutor(
                max_workers=num_workers, initializer=_worker_init
            ) as executor:
                futures = {}
                
                for params in combinations:
                    ai, _, chord, sol, vinf = params
                    airfoil_name = config["solver"]["neuralfoil"]["airfoil"][ai]
                    
                    # Subpasta individual para cada caso da varredura
                    case_dir = sweep_base_dir / f"{airfoil_name}_c{chord:.2f}_sol{sol:.3f}_v{vinf:.2f}"

                    future = executor.submit(
                        run_simulation_case, 
                        params, 
                        config, 
                        flow_cfg, 
                        stall_angles, 
                        output_dir=case_dir,  # Garante pasta própria para este caso
                        show_details=False
                    )
                    futures[future] = params

                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        params = futures.get(future)
                        result = {
                            "name": str(params),
                            "status": "ERROR",
                            "time_sec": 0.0,
                            "error": repr(e),
                        }

                    results.append(result)
                    completed += 1

                    now = time.perf_counter()
                    if (now - last_update > 0.05) or (completed == total):
                        UI.progress_bar(
                            completed, total, now - start_time, prefix="2D Sweep"
                        )
                        last_update = now

        print()
    # =========================================================================
    # 4. EXPORT & SUMMARY
    # =========================================================================
    UI.section("EXPORT & SUMMARY")

    log_path = export_2d_results(results, config)
    total_time = time.perf_counter() - start_time

    UI.status("Sweep Execution Time", f"{total_time:.2f} s")
    UI.status("Results Log", log_path, level="ok")
    print()


if __name__ == "__main__":
    run_simulation()

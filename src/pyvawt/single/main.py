import os
import csv
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pyvawt.single.simulation import run_simulation_case, simulate_3D_turbine, warmup_numba_kernels, _worker_init
from src.pyvawt.single.utils import (
    load_config,
    detect_stall_angles,
    print_config,
    print_summary,
    parse_args,
    print_simulation_footer,
    print_simulation_results,
    format_time,
    export_3d_results,
    export_2d_results
)

from src.pyvawt.ui.ui import UI

#run command: uv run python3 -m src.pyvawt.single.main
# test/data/config.yaml src/pyvawt/config/config.yaml

def run_simulation(config_path: str = 'src/pyvawt/config/config.yaml'):
    """
    Runs a batch of aerodynamic simulations defined by the configuration file.
    Handles both 3D multi-slice mode and 2D parallel sweep mode.
    """
    # 1. Header & Initialization
    UI.banner("PYVAWT - AERODYNAMIC SIMULATOR")

    UI.section("INITIALIZATION & PRE-PROCESSING")
    warmup_numba_kernels(verbose=True)

    args = parse_args()
    config = load_config(path=args.config or config_path)

    # Compute stall angles for all configured airfoils
    stall_angles = {}
    for airfoil_index in range(len(config['solver']['neuralfoil']['airfoil'])):
        aoaStallPos, aoaStallNeg = detect_stall_angles(config, airfoil_index)
        stall_angles[airfoil_index] = (aoaStallPos, aoaStallNeg)

    # Optional configuration printout
    if args.show_config:
        print_config(config)

    # =========================================================================
    # 2. CHECK 3D SIMULATION MODE
    # =========================================================================
    sim3d_cfg = config.get('solver', {}).get('simulation3d', {})

    if sim3d_cfg.get('enabled', False):
        res_3d = simulate_3D_turbine(config, stall_angles)
        
        export_3d_results(
            tsr=res_3d['tsr'], 
            cp_3d=res_3d['cp_3d'], 
            config=config, 
            output_dir=res_3d['result_dir']
        )
        return

    # =========================================================================
    # 3. 2D PARALLEL SWEEP MODE
    # =========================================================================
    UI.section("EXECUTION (2D SWEEP)")

    airfoil_indices = list(range(len(config['solver']['neuralfoil']['airfoil'])))
    chords = config['turbine']['chord']
    solidities = config['turbine']['solidity']
    vinfs = config['environment']['Vinf'] 
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

    # --- OTIMIZAÇÃO: CASO ÚNICO (Sem overhead de ProcessPool) ---
    if total == 1:
        UI.status("Simulation Cases", "1 combination (Single-Core Direct Execution)")
        params = combinations[0]
        result = run_simulation_case(params, config, flow_cfg, stall_angles)
        results.append(result)
        UI.progress_bar(1, 1, time.perf_counter() - start_time, prefix="2D Sweep")

    # --- VARREDURA MULTI-CORE (Throttled Progress Bar) ---
    else:
        num_workers = min(total, os.cpu_count())
        UI.status("Simulation Cases", f"{total} combinations")
        UI.status("Active CPU Cores", f"{num_workers}")
        print()

        completed = 0
        last_update = 0.0  # Controle de tempo para limite de FPS do terminal

        with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as executor:
            futures = {
                executor.submit(run_simulation_case, params, config, flow_cfg, stall_angles): params
                for params in combinations
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    params = futures.get(future)
                    result = {'name': str(params), 'status': 'ERROR', 'time_sec': 0.0, 'error': repr(e)}
                
                results.append(result)
                completed += 1

                # Atualiza a barra no máximo a cada 0.05s (20 FPS) ou na conclusão
                now = time.perf_counter()
                if (now - last_update > 0.05) or (completed == total):
                    UI.progress_bar(completed, total, now - start_time, prefix="2D Sweep")
                    last_update = now
    # =========================================================================
    # 4. EXPORT & SUMMARY
    # =========================================================================
    UI.section("EXPORT & SUMMARY")
    
    log_path = export_2d_results(results, config)
    total_time = time.perf_counter() - start_time

    UI.status("Sweep Execution Time", f"{total_time:.2f} s")
    UI.status("Results Log", log_path, level="ok")
    print()

    # Detailed summary breakdown
    # print_simulation_results(results, start_time, log_path)


if __name__ == '__main__':
    run_simulation()

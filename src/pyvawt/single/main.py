import os
import csv
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pyvawt.single.simulation import run_simulation_case, simulate_3D_turbine
from src.pyvawt.single.utils import (
    load_config,
    detect_stall_angles,
    print_config,
    print_summary,
    parse_args,
    print_simulation_footer,
    print_simulation_results,
    format_time,
)
#run command: uv run python3 -m src.pyvawt.single.main
# test/data/config.yaml src/pyvawt/config/config.yaml

def run_simulation(config_path: str = 'src/pyvawt/config/config.yaml'):
    '''
    Run a batch of aerodynamic simulations defined by the configuration.
    '''
    args = parse_args()
    config = load_config(path=args.config or 'src/pyvawt/config/config.yaml')

    stall_angles = {}

    for airfoil_index in range(len(config['solver']['neuralfoil']['airfoil'])):
        aoaStallPos, aoaStallNeg = detect_stall_angles(config, airfoil_index)
        stall_angles[airfoil_index] = (aoaStallPos, aoaStallNeg)

    print("Airfoil stall angles computed successfully.\n")

    # ==========================
    # CHECK 3D SIMULATION MODE
    # ==========================
    sim3d_cfg = config.get('solver', {}).get('simulation3d', {})

    if sim3d_cfg.get('enabled', False):
        print("\n==============================")
        print("3D simulation mode ENABLED")
        print("==============================\n")

        simulate_3D_turbine(config, stall_angles)

        print("\n3D simulation finished.\n")

        return

    print_summary(config)

    if args.show_config:
        print("\nFull configuration")
        print("=" * 40)
        print_config(config)

    # Carregamento dos vetores de varredura (Sweep)
    airfoil_indices = list(range(len(config['solver']['neuralfoil']['airfoil'])))
    chords = config['turbine']['chord']
    solidities = config['turbine']['solidity']
    vinfs = config['environment']['Vinf'] 
    flow_cfg = config.get("submodels", {}).get("flow_curvature", {})

    # Montagem das combinações
    combinations = []
    for ai in airfoil_indices:
        for chord in chords:
            for sol in solidities:
                for vinf in vinfs:
                    combinations.append((ai, 0, chord, sol, vinf))

    total = len(combinations)
    if total > 1:
        print(f'Initiating {total} parallel simulation cases...\n')
    else:
        print(f'Initiating {total} parallel simulation case...\n')

    results = []
    start_time = time.time()
    completed = 0
    UPDATE_EVERY = max(1, total // 100)  # controls update frequency

    # Simple progress bar with low computational cost
    def _print_progress(completed, total, start_time):
        elapsed = time.time() - start_time
        pct = (completed / total) * 100 if total else 100.0

        elapsed_m = int(elapsed // 60)
        elapsed_s = int(elapsed % 60)

        if completed > 0 and total > completed:
            avg_time = elapsed / completed
            eta = avg_time * (total - completed)
            eta_m = int(eta // 60)
            eta_s = int(eta % 60)
            eta_str = f"{eta_m:02d}:{eta_s:02d}"
        else:
            eta_str = "00:00"

        if total > 1:
            msg = (
                f"\rProgress: {completed}/{total} cases ({pct:5.1f}%) "
                f"| Elapsed {elapsed_m:02d}:{elapsed_s:02d} "
                f"| ETA {eta_str}"
            )
        else:
            msg = (
                f"\rProgress: {completed}/{total} case ({pct:5.1f}%) "
                f"| Elapsed {elapsed_m:02d}:{elapsed_s:02d}"
            )

        sys.stdout.write(msg)
        sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(run_simulation_case, params, config, flow_cfg, stall_angles): params
            for params in combinations
        }

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                params = futures.get(future)
                name = str(params)
                result = {
                    'name': name,
                    'status': 'ERROR',
                    'time_sec': 0.0,
                    'error': repr(e)
                }
            results.append(result)
            completed += 1
            if (completed % UPDATE_EVERY) == 0 or completed == total:
                _print_progress(completed, total, start_time)

    print()

    log_path = os.path.join('src/results/temporary_results', 'log_simulacoes.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print_simulation_results(results, start_time, log_path)

if __name__ == '__main__':
    run_simulation()

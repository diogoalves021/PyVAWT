import os
import csv
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pyvawt.simulation import run_simulation_case
from src.pyvawt.utils import (
    load_config,
    detect_stall_angles,
    print_config,
    print_summary,
    parse_args,
    print_simulation_footer,
    print_simulation_results,
    format_time_sec_to_minsec,
)
#run command: uv run python3 -m src.pyvawt.main
# test/data/config.yaml src/pyvawt/config/config.yaml

def run_simulation(config_path: str = 'src/pyvawt/config/config.yaml'):
    '''
    Run a batch of aerodynamic simulations defined by the configuration.

    The function loads the configuration via :func:`load_config`, enumerates all
    parameter combinations (airfoil, turbine index, chord, solidity, Vinf) and
    executes each simulation in parallel using :class:`concurrent.futures.ProcessPoolExecutor`.
    Results are collected in memory, written to a CSV file under
    ``src/results/temporary_results/log_simulacoes.csv``, and a light textual
    summary table is printed to stdout using :func:`print_summary_tabulate`.

    Notes
    -----
    - Each simulation case is executed by :func:`run_simulation_case`.
    - All tasks are submitted to the executor at once. If the total number of
      combinations is very large this can increase memory usage; consider using
      a generator, batching, or a streaming submission strategy in that case.
    - Progress output is a minimal ASCII counter updated periodically to reduce
      terminal rendering overhead (``UPDATE_EVERY = max(1, total // 100)``).
    - If your simulation code uses multi-threaded native libraries (e.g. BLAS,
      OpenMP), set environment variables such as ``OMP_NUM_THREADS=1``,
      ``OPENBLAS_NUM_THREADS=1`` and ``MKL_NUM_THREADS=1`` before creating the pool
      to avoid oversubscription.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return a value. Results are written to disk and a
        summary is printed to stdout.

    Side effects
    ------------
    - Creates directories and files under ``src/results/temporary_results/`` (one
      folder per case when enabled in the config) and writes a CSV summary at
      ``src/results/temporary_results/log_simulacoes.csv``.
    - Prints progress messages and the final summary table to stdout.
    - Exceptions raised inside individual simulation cases are captured and
      recorded as result rows with ``status='ERROR'`` and an ``error`` field.

    Raises
    ------
    IndexError
        If no combinations are generated, the code currently assumes at least
        one result exists and may attempt to access ``results[0]`` when writing
        the CSV. Consider adding an explicit check for an empty results list.
    OSError
        If creating the output directory or writing files fails (e.g. due to
        permissions), an OSError may be raised.

    Examples
    --------
    Run all simulations defined by the current configuration (blocking call):

    >>> run_simulation()
    '''
    args = parse_args()
    config = load_config(path=args.config or 'src/pyvawt/config/config.yaml')
    # config = load_config(path=config_path)
    print_summary(config)

    if args.show_config:
        print("\nFull configuration")
        print("=" * 40)
        print_config(config)

    stall_angles = {}

    for airfoil_index in range(len(config['simulation']['airfoil'])):
        aoaStallPos, aoaStallNeg = detect_stall_angles(config, airfoil_index)
        stall_angles[airfoil_index] = (aoaStallPos, aoaStallNeg)

    print("Airfoil stall angles computed successfully.\n")

    airfoil_indices = list(range(len(config['simulation']['airfoil'])))
    turbine_indices = list(range(config['simulation']['num_turbines']))
    chords = config['turbine']['chord']
    solidities = config['turbine']['solidity']
    vinfs = config['environment']['Vinf']
    flow_cfg = config.get("flow_curvature", {})

    combinations = []
    for ai in airfoil_indices:
        for ti in turbine_indices:
            for chord in chords:
                for sol in solidities:
                    for vinf in vinfs:
                        combinations.append((ai, ti, chord, sol, vinf))

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

        # ---- ETA calculation ----
        if completed > 0 and total > completed:
            avg_time = elapsed / completed
            eta = avg_time * (total - completed)

            eta_m = int(eta // 60)
            eta_s = int(eta % 60)
            eta_str = f"{eta_m:02d}:{eta_s:02d}"
        else:
            eta_str = "00:00"

        # ---- message ----
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
            # Updates every N completions for time saving
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

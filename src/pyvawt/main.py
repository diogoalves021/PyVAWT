import os
import csv
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tabulate import tabulate

from src.pyvawt.simulation import run_simulation_case
from src.pyvawt.utils import load_config

#run command: uv run python3 -m src.pyvawt.main

def format_time_sec_to_minsec(seconds):
    try:
        seconds = float(seconds)
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{minutes:02d}:{secs:02d}'
    except Exception:
        return str(seconds)

def print_summary_tabulate(results, log_path):
    if not results:
        print(f'No results to show. Log saved to: {log_path}')
        return

    headers = ['Case', 'Status', 'Time (MM:SS)']
    rows = []
    for r in results:
        time_val = format_time_sec_to_minsec(r.get('time_sec', ''))
        rows.append((r.get('name', ''), r.get('status', ''), time_val))

    print()  # garantir separação do progresso
    print(tabulate(rows, headers=headers, tablefmt='plain'))
    print(f'\nLog saved to: {log_path}')

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
    config = load_config(path=config_path)
    print(f'DEBUG (run_simulation) - config: {config}')
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
    print(f'Initiating {total} parallel simulation cases...\n')

    results = []
    start_time = time.time()
    completed = 0
    UPDATE_EVERY = max(1, total // 100)  # controls update frequency

    # Simple progress bar with low computational cost
    def _print_progress(completed, total, start_time):
        elapsed = time.time() - start_time
        pct = (completed / total) * 100 if total else 100.0
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        sys.stdout.write(f'\rSimulating cases: {completed}/{total} ({pct:5.1f}%) Elapsed {mins:02d}:{secs:02d}')
        sys.stdout.flush()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(run_simulation_case, params, config, flow_cfg): params
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
    print_summary_tabulate(results, log_path)


if __name__ == '__main__':
    run_simulation()
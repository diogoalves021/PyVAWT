import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn
from rich.console import Console
from rich.table import Table
from rich import box
from src.pyvawt.simulation import run_simulation_case
from src.pyvawt.utils import load_config

def run_simulation():
    """
    Runs a batch of aerodynamic simulations for different parameter combinations.

    The function defines a small set of test cases using airfoil, chord, solidity,
    and freestream velocity values, and runs them in parallel using multiple CPU cores.

    Results are saved to disk:
    - Individual simulation results are stored in subfolders under 'src/results/temporary_results'.
    - A log file named 'log_simulacoes.csv' is saved summarizing all simulations.

    Notes
    -----
    - Uses ProcessPoolExecutor for parallel execution of simulations.
    - The list of parameters can be modified directly within the function.
    - This is the main function to be run as a script.
    """
    config = load_config()

    airfoil_indices = list(range(len(config["simulation"]["airfoil"])))
    turbine_indices = list(range(config["simulation"]["num_turbines"]))
    chords = config["turbine"]["chord"]
    solidities = config["turbine"]["solidity"]
    vinfs = config["environment"]["Vinf"]

    combinations = []
    for ai in airfoil_indices:
        for ti in turbine_indices:
            for chord in chords:
                for sol in solidities:
                    for vinf in vinfs:
                        combinations.append((ai, ti, chord, sol, vinf))

    print(f"Initiating {len(combinations)} parallel simulation cases...\n")

    console = Console()
    results = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Simulating cases...", total=len(combinations))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(run_simulation_case, params, config): params
                for params in combinations
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.advance(task)

    log_path = os.path.join("src/results/temporary_results", "log_simulacoes.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    table = Table(title="Simulation Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Case", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Time (s)", justify="right")

    for res in results:
        status_color = "green" if res["status"] == "OK" else "red"
        table.add_row(res["name"], f"[{status_color}]{res['status']}[/{status_color}]", str(res["time_sec"]))

    console.print(table)
    console.print(f"\n[bold green]Log saved to:[/bold green] {log_path}")


if __name__ == "__main__":
    run_simulation()


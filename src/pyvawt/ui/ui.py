"""
Terminal User Interface (UI) utilities for VAWT simulations.

Provides ANSI formatting helper classes, progress bar renderers, status indicators,
and structured text tables for single- and multi-turbine aerodynamic runs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from tabulate import tabulate


# ==============================================================================
# TERMINAL UI HELPERS
# ==============================================================================

class UI:
    """
    Terminal User Interface helper providing ANSI styling, status indicators, and progress bars.

    Attributes
    ----------
    RESET : str
        ANSI code to reset text formatting.
    BOLD : str
        ANSI code for bold text style.
    DIM : str
        ANSI code for dimmed/faded text style.
    CYAN : str
        ANSI code for cyan color output.
    GREEN : str
        ANSI code for green color output.
    YELLOW : str
        ANSI code for yellow color output.
    RED : str
        ANSI code for red color output.
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"

    @staticmethod
    def banner(title: str) -> None:
        """
        Display a centered main header banner enclosed in a styled box.

        Parameters
        ----------
        title : str
            Title string to center inside the box header.
        """
        line = "═" * 68
        print(f"\n{UI.CYAN}{UI.BOLD}╔{line}╗")
        print(f"║ {title.center(66)} ║")
        print(f"╚{line}╝{UI.RESET}\n")

    @staticmethod
    def section(title: str) -> None:
        """
        Display an unnumbered section divider in the terminal.

        Parameters
        ----------
        title : str
            Section header title text.
        """
        print(f"\n{UI.BOLD}{UI.CYAN}─── {title} {"─" * (60 - len(title))}{UI.RESET}")

    @staticmethod
    def status(key: str, value: str, level: str = "info") -> None:
        """
        Display an aligned key-value status row with level-based color highlights.

        Parameters
        ----------
        key : str
            Label describing the metric, parameter, or status field.
        value : str
            Value or descriptive message to display.
        level : {"info", "ok", "warn"}, default="info"
            Category determining the value string color:
            - "ok": Green
            - "warn": Yellow
            - "info" (or default): Standard color
        """
        color = UI.GREEN if level == "ok" else UI.YELLOW if level == "warn" else UI.RESET
        print(f"  {UI.DIM}•{UI.RESET} {key:<28} : {color}{value}{UI.RESET}")

    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format execution time with sub-second precision.

        Parameters
        ----------
        seconds : float
            Time duration in seconds.

        Returns
        -------
        str
            Formatted time string (e.g., "0.89s" or "1m 12.45s").
        """
        if seconds <= 0:
            return "0.00s"
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = int(seconds // 60)
        rem_sec = seconds % 60
        return f"{minutes}m {rem_sec:.2f}s"

    @staticmethod
    def progress_bar(
        current: int, total: int, elapsed_sec: float, prefix: str = "Progress"
    ) -> None:
        """
        Render an in-place single-line graphical progress bar with ETA and elapsed time.

        Parameters
        ----------
        current : int
            Number of completed iterations or slices.
        total : int
            Total target iterations or slices.
        elapsed_sec : float
            Elapsed execution time in seconds.
        prefix : str, default="Progress"
            Label displayed prior to the progress bar.
        """
        percent = (current / total) * 100 if total > 0 else 100.0
        bar_len = 25
        filled = int(bar_len * current // total) if total > 0 else bar_len
        bar = "█" * filled + "░" * (bar_len - filled)

        if 0 < current < total:
            eta_sec = (elapsed_sec / current) * (total - current)
            eta_str = UI.format_time(eta_sec)
        else:
            eta_str = "0.00s"

        elapsed_str = UI.format_time(elapsed_sec)

        msg = (
            f"\r  {UI.CYAN}{prefix}{UI.RESET} [{UI.GREEN}{bar}{UI.RESET}] "
            f"{UI.BOLD}{percent:5.1f}%{UI.RESET} ({current}/{total}) "
            f"| {UI.DIM}Elapsed:{UI.RESET} {elapsed_str} "
            f"| {UI.DIM}ETA:{UI.RESET} {eta_str}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
        if current == total:
            print()

class MultiTurbineUI:
    """
    Terminal UI helper tailored for multi-turbine array simulations.
    """

    @staticmethod
    def print_header() -> None:
        """
        Display the main banner header for multi-turbine simulations.
        """
        UI.banner("PYVAWT - MULTI-TURBINE SOLVER MODULE")

    @staticmethod
    def print_init(
        turbines: list[Any], jit_time: float, mode_coupled: bool = True
    ) -> None:
        """
        Display initialization status, solver mode, and individual turbine configurations.

        Parameters
        ----------
        turbines : list of Any
            List of turbine objects present in the simulated array.
        jit_time : float
            Time elapsed during JIT kernel compilation or warmup [s].
        mode_coupled : bool, default=True
            Whether the multi-turbine interaction solver is enabled.
        """
        UI.section("INITIALIZATION & SYSTEM CONFIGURATION")
        UI.status(
            "JIT Engine (Numba)", f"Ready ({UI.format_time(jit_time)})", level="ok"
        )

        mode_str = "ENABLED" if mode_coupled else "DISABLED"
        UI.status("Coupled Solver Mode", mode_str)
        UI.status("Turbine Array Count", f"{len(turbines)} Turbines")

        for i, t in enumerate(turbines):
            is_last = i == len(turbines) - 1
            tree_char = "└──" if is_last else "├──"
            pos_info = (
                f"Pos ({t.centerX:.1f}, {t.centerY:.1f}) m  | R = {t.r:.2f} m"
            )
            print(
                f"    {UI.DIM}{tree_char}{UI.RESET} Turbine {i+1:<17}: {UI.RESET}{pos_info}"
            )
        print()

    @staticmethod
    def print_progress(current: int, total: int, elapsed_sec: float) -> None:
        """
        Render the progress bar for a coupled multi-turbine sweep.

        Parameters
        ----------
        current : int
            Current completed iteration or step count.
        total : int
            Total target iterations or step count.
        elapsed_sec : float
            Elapsed execution time [s].
        """
        UI.progress_bar(current, total, elapsed_sec, prefix="Coupled Sweep")

    @staticmethod
    def print_results(
        turbines: list[Any],
        total_time: float,
        output_dir: str | Path,
        cp_results: Any = None,
        ct_results: Any = None,
        tsr_vec: Any = None,
        warnings: list[str] | None = None,
    ) -> None:
        """
        Display the final multi-turbine summary table and convergence warnings.

        Parameters
        ----------
        turbines : list of Any
            List of turbine objects evaluated during the run.
        total_time : float
            Total simulation execution time [s].
        output_dir : str or Path
            Path to the directory where output artifacts were exported.
        cp_results : Any, optional
            Power coefficient results matrix or data structure.
        ct_results : Any, optional
            Thrust coefficient results matrix or data structure.
        tsr_vec : Any, optional
            Vector of tip speed ratios evaluated during the sweep.
        warnings : list of str, optional
            List of warnings or convergence messages logged during simulation.
        """
        if warnings is None:
            warnings = []

        UI.section("MULTI-TURBINE SIMULATION RESULTS")

        has_warnings = len(warnings) > 0
        status_msg = (
            "Completed Successfully"
            if not has_warnings
            else f"Completed with Warnings ({len(warnings)} Issue{'s' if len(warnings) > 1 else ''})"
        )
        status_level = "warn" if has_warnings else "ok"

        time_str = f"{UI.format_time(total_time)}"

        UI.status("Status", status_msg, level=status_level)
        UI.status("Total Execution Time", time_str)
        UI.status("Output Directory", str(output_dir))
        print()

        print(f"  {UI.BOLD}Turbine Array Summary:{UI.RESET}")
        print(f"  {UI.CYAN}┌───────────┬────────────────┬────────────┐{UI.RESET}")
        print(
            f"  {UI.CYAN}│{UI.RESET} Turbine   {UI.CYAN}│{UI.RESET} Position (X,Y) {UI.CYAN}│{UI.RESET} Status     {UI.CYAN}│{UI.RESET}"
        )
        print(f"  {UI.CYAN}├───────────┼────────────────┼────────────┤{UI.RESET}")

        for i, t in enumerate(turbines):
            pos_str = f"({t.centerX:.1f}, {t.centerY:.1f}) m"

            t_has_warn = any(
                f"Turbine {i+1}" in w or f"turbina {i+1}" in w.lower()
                for w in warnings
            )
            t_status_str = (
                f"{UI.YELLOW}Warning{UI.RESET}"
                if t_has_warn
                else f"{UI.GREEN}OK{UI.RESET}"
            )

            print(
                f"  {UI.CYAN}│{UI.RESET} Turbine {i+1:<1} {UI.CYAN}│{UI.RESET} {pos_str:<14} {UI.CYAN}│{UI.RESET} {t_status_str:<19} {UI.CYAN}│{UI.RESET}"
            )

        print(f"  {UI.CYAN}└───────────┴────────────────┴────────────┘{UI.RESET}\n")

        if warnings:
            print(f"  {UI.YELLOW}{UI.BOLD}Convergence Warnings Breakdown:{UI.RESET}")
            for w in warnings:
                print(f"    {UI.DIM}•{UI.RESET} {w}")
            print()


# ==============================================================================
# REPORTING & FORMATTING HELPERS
# ==============================================================================

def format_time(seconds: float | str) -> str:
    """
    Format time duration in seconds into a formatted MM:SS string.

    Parameters
    ----------
    seconds : float or str
        Time duration in seconds.

    Returns
    -------
    str
        Formatted time string (e.g., '02:15') or string representation of input
        if conversion fails.
    """
    try:
        sec_float = float(seconds)
        minutes = int(sec_float // 60)
        secs = int(sec_float % 60)
        return f"{minutes:02d}:{secs:02d}"
    except (ValueError, TypeError):
        return str(seconds)


def _format_value(value: Any) -> str:
    """Format configuration primitive values into clean terminal strings."""
    if isinstance(value, bool):
        return "Enabled" if value else "Disabled"
    if isinstance(value, list) and len(value) == 1:
        return str(value[0])
    if isinstance(value, dict):
        # Format inline dictionary without raw brackets/quotes
        formatted_items = [
            f"{k}={_format_value(v)}"
            for k, v in value.items()
            if not isinstance(v, dict)
        ]
        return ", ".join(formatted_items) if formatted_items else str(value)
    return str(value)


def print_simulation_config(config: dict[str, Any]) -> None:
    """
    Print simulation configuration in a clean, standardized CLI box style.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary containing turbine, environment, solver,
        submodels, and output sections.
    """
    print("\n─── SIMULATION CONFIGURATION ────────────────────────────────────\n")

    section_labels: dict[str, str] = {
        "turbine": "TURBINE",
        "environment": "ENVIRONMENT",
        "solver": "SOLVER",
        "submodels": "SUBMODELS",
        "output": "OUTPUT SETTINGS",
    }

    key_aliases: dict[str, str] = {
        # Turbine
        "r": "Radius (r)",
        "height": "Height",
        "twist": "Twist Angle",
        "delta": "Cone/Inclin. Angle (delta)",
        "chord": "Blade Chord",
        "B": "Number of Blades (B)",
        "solidity": "Solidity",
        "centerX": "Center X",
        "centerY": "Center Y",
        "Omega": "Rotational Speed (Omega)",
        "ntheta": "Azimuthal Discretization",
        # Environment
        "Vinf": "Freestream Velocity (Vinf)",
        "rho": "Air Density (rho)",
        "mu": "Dynamic Viscosity (mu)",
        # Solver
        "tsr": "TSR Range",
        "method": "Polar Generator Method",
        "fixed_parameter": "Fixed Parameter",
        "neuralfoil": "NeuralFoil Settings",
        "file": "Data File Source",
        "simulation3d": "3D Simulation Setup",
        # Submodels
        "tip_loss": "Tip Loss Correction",
        "dynamic_stall": "Dynamic Stall Model",
        "flow_curvature": "Flow Curvature Model",
        # Output
        "save": "Save Data Output",
        "save_config": "Save Config File",
        "save_plot": "Save Plots",
        "data_file": "Data Export Format",
        "plot_image": "Plot Resolution/Format",
        "cp_theta": "Cp-Theta Analysis",
    }

    for sec_key, sec_data in config.items():
        if not isinstance(sec_data, dict):
            continue

        title = section_labels.get(sec_key, sec_key.upper())
        print(f"  [{title}]")

        for key, val in sec_data.items():
            label = key_aliases.get(key, key)
            formatted_val = _format_value(val)
            print(f"  • {label:<30} : {formatted_val}")

        print()  # Spacer line between sections


def print_summary(config: dict[str, Any]) -> None:
    """
    Print formatted summary of simulation, fluid properties, and reference quantities.

    Calculates reference non-dimensional flow numbers (Reynolds and Mach) based
    on configuration entries.

    Parameters
    ----------
    config : dict
        Simulation configuration dictionary containing turbine, environment,
        solver, and submodel definitions.
    """
    print("\nSimulation summary")
    print("=" * 40)

    turb = config.get("turbine", {})
    env = config.get("environment", {})
    solver = config.get("solver", {})
    submodels = config.get("submodels", {})

    def _get_ref(val: Any) -> Any:
        return val[0] if isinstance(val, (list, tuple)) else val

    vinf_ref = _get_ref(env.get("Vinf", 0.0))
    chord_ref = _get_ref(turb.get("chord", 0.0))
    rho = env.get("rho", 1.225)
    mu = env.get("mu", 1.789e-5)
    a = env.get("speed_of_sound", 343.0)

    re_calc = (rho * vinf_ref * chord_ref) / mu if mu else 0.0
    mach_calc = vinf_ref / a if a else 0.0

    print("\nTURBINE")
    print("-" * 40)
    print(f"radius (r)           : {turb.get('r')}")
    print(f"height (H)           : {turb.get('H')}")
    print(f"blades (B)           : {turb.get('B')}")
    print(f"chord                : {turb.get('chord')}")
    print(f"theta discretization : {turb.get('ntheta')}")

    print("\nENVIRONMENT")
    print("-" * 40)
    print(f"wind speed (Vinf)    : {env.get('Vinf')}")
    print(f"density (rho)        : {env.get('rho')}")

    print("\nSIMULATION")
    print("-" * 40)
    airfoil_ref = solver.get("neuralfoil", {}).get("airfoil")
    print(f"airfoil              : {airfoil_ref}")
    print(f"Reynolds number (Ref): {re_calc:.2e}")
    print(f"Mach number (Ref)    : {mach_calc:.3f}")
    print(f"dynamic stall        : {submodels.get('dynamic_stall')}")

    print("=" * 40)


def print_simulation_results(
    results: list[dict[str, Any]], start_time: float, log_path: str | Path
) -> None:
    """
    Print final simulation case results table and overall execution statistics.

    Parameters
    ----------
    results : list of dict
        List of execution dictionaries containing case metrics, status, and runtimes.
    start_time : float
        Initial time recorded at the start of the simulation run [s].
    log_path : str or Path
        Path to the output log file.
    """
    print("\n" + "=" * 40)
    print("SIMULATION RESULTS")
    print("=" * 40)

    if not results:
        print("No results to show.")
        print(f"Log file : {log_path}")
        print("=" * 40)
        return

    headers = ["Case", "Status", "Time (MM:SS)"]
    rows = []

    for r in results:
        time_val = format_time(r.get("time_sec", ""))
        rows.append((r.get("name", ""), r.get("status", ""), time_val))

    print()
    print(tabulate(rows, headers=headers, tablefmt="plain"))

    total_cases = len(results)
    successful = sum(1 for r in results if r.get("status") == "OK")
    failed = total_cases - successful

    total_time = time.time() - start_time
    mins = int(total_time // 60)
    secs = int(total_time % 60)

    avg_time = total_time / total_cases if total_cases else 0
    avg_mins = int(avg_time // 60)
    avg_secs = int(avg_time % 60)

    print("-" * 40)
    print(f"Cases executed : {total_cases}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {failed}")
    print()
    print(f"Total runtime  : {mins:02d}:{secs:02d}")
    print(f"Average case   : {avg_mins:02d}:{avg_secs:02d}")
    print()
    print(f"Log file       : {log_path}")
    print("=" * 40)


def print_simulation_footer(
    results: list[dict[str, Any]], start_time: float, log_path: str | Path
) -> None:
    """
    Print clear final summary footer of the simulation run.

    Parameters
    ----------
    results : list of dict
        List of execution dictionaries containing case metrics, status, and runtimes.
    start_time : float
        Initial time recorded at the start of the simulation run [s].
    log_path : str or Path
        Path to the output log file.
    """
    print_simulation_results(results, start_time, log_path)

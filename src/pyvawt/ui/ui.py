import sys
from typing import List, Optional

# Terminal UI 
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
    @staticmethod
    def print_header():
        UI.banner("PYVAWT - MULTI-TURBINE AERODYNAMIC SIMULATOR")

    @staticmethod
    def print_init(turbines, jit_time: float, mode_coupled: bool = True):
        UI.section("INITIALIZATION & SYSTEM CONFIGURATION")
        UI.status("JIT Engine (Numba)", f"Ready ({UI.format_time(jit_time)})", level="ok")
        
        mode_str = "ENABLED" if mode_coupled else "DISABLED"
        UI.status("Coupled Solver Mode", mode_str)
        UI.status("Turbine Array Count", f"{len(turbines)} Turbines")
        
        # Árvore de detalhamento individual de cada turbina
        for i, t in enumerate(turbines):
            is_last = (i == len(turbines) - 1)
            tree_char = "└──" if is_last else "├──"
            pos_info = f"Pos ({t.centerX:.1f}, {t.centerY:.1f}) m  | R = {t.r:.2f} m"
            print(f"    {UI.DIM}{tree_char}{UI.RESET} Turbine {i+1:<17}: {UI.RESET}{pos_info}")
        print()

    @staticmethod
    def print_progress(current: int, total: int, elapsed_sec: float):
        UI.progress_bar(current, total, elapsed_sec, prefix="Coupled Sweep")

    @staticmethod
    def print_results(
        turbines, 
        total_time: float, 
        output_dir: str, 
        cp_results=None, 
        ct_results=None, 
        tsr_vec=None, 
        warnings: Optional[List[str]] = None
    ):
        if warnings is None:
            warnings = []

        UI.section("MULTI-TURBINE SIMULATION RESULTS")
        
        has_warnings = len(warnings) > 0
        status_msg = "Completed Successfully" if not has_warnings else f"Completed with Warnings ({len(warnings)} Issue{'s' if len(warnings)>1 else ''})"
        status_level = "warn" if has_warnings else "ok"
        
        time_str = f"{UI.format_time(total_time)}"
        
        UI.status("Status", status_msg, level=status_level)
        UI.status("Total Execution Time", time_str)
        UI.status("Output Directory", output_dir)
        print()

        # Tabela resumo simplificada do parque (Sem Cp e Ct)
        print(f"  {UI.BOLD}Turbine Array Summary:{UI.RESET}")
        print(f"  {UI.CYAN}┌───────────┬────────────────┬────────────┐{UI.RESET}")
        print(f"  {UI.CYAN}│{UI.RESET} Turbine   {UI.CYAN}│{UI.RESET} Position (X,Y) {UI.CYAN}│{UI.RESET} Status     {UI.CYAN}│{UI.RESET}")
        print(f"  {UI.CYAN}├───────────┼────────────────┼────────────┤{UI.RESET}")

        for i, t in enumerate(turbines):
            pos_str = f"({t.centerX:.1f}, {t.centerY:.1f}) m"
            
            # Verifica se houve warning para essa turbina específica
            t_has_warn = any(f"Turbine {i+1}" in w or f"turbina {i+1}" in w.lower() for w in warnings)
            t_status_str = f"{UI.YELLOW}Warning{UI.RESET}" if t_has_warn else f"{UI.GREEN}OK{UI.RESET}"
            
            print(f"  {UI.CYAN}│{UI.RESET} Turbine {i+1:<1} {UI.CYAN}│{UI.RESET} {pos_str:<14} {UI.CYAN}│{UI.RESET} {t_status_str:<19} {UI.CYAN}│{UI.RESET}")

        print(f"  {UI.CYAN}└───────────┴────────────────┴────────────┘{UI.RESET}\n")

        if warnings:
            print(f"  {UI.YELLOW}{UI.BOLD}Convergence Warnings Breakdown:{UI.RESET}")
            for w in warnings:
                print(f"    {UI.DIM}•{UI.RESET} {w}")
            print()

"""
Main Entrypoint Module for PyVAWT.

Provides the primary Command Line Interface (CLI) execution routine for running
coupled Vertical Axis Wind Turbine (VAWT) aerodynamic simulations.
"""
from __future__ import annotations

import logging

from src.pyvawt.multiple.orchestrator import run_simulation_case
from src.pyvawt.multiple.settings import load_config
from src.pyvawt.ui.ui import MultiTurbineUI

# Application-level logging configuration fallback
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """
    Execute the primary PyVAWT simulation pipeline.

    Loads simulation configuration, extracts base physical parameters, initializes
    the Terminal User Interface (UI), and triggers the orchestrator execution
    routine for the configured simulation case.

    Raises
    ------
    Exception
        Logs any unhandled critical exception raised during simulation setup,
        numerical calculation, or results exporting.
    """
    # 1. Initialize Terminal UI Header
    MultiTurbineUI.print_header()

    try:
        # 2. Load cached YAML configuration dictionary
        config = load_config()

        # 3. Safely extract base physical and operational parameters
        raw_chord = config["turbine"]["chord"]
        raw_solidity = config["turbine"]["solidity"]
        raw_vinf = config["environment"]["Vinf"]

        chord_val = float(raw_chord[0] if isinstance(raw_chord, list) else raw_chord)
        solidity_val = float(raw_solidity[0] if isinstance(raw_solidity, list) else raw_solidity)
        vinf_val = float(raw_vinf[0] if isinstance(raw_vinf, list) else raw_vinf)

        # 4. Pack case parameters tuple: (case_idx, total_cases, chord, solidity, vinf)
        case_params = (1, 1, chord_val, solidity_val, vinf_val)

        # 5. Dispatch case run to orchestrator
        run_simulation_case(case_params)

    except Exception as err:
        logger.critical(f"Fatal error during simulation execution: {err}", exc_info=True)


if __name__ == "__main__":
    main()

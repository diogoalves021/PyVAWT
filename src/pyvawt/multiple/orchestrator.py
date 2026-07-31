"""
Simulation Orchestration Module for PyVAWT.

Provides high-level execution routines for Vertical Axis Wind Turbine (VAWT)
simulations, including physical layout initialization, Tip Speed Ratio (TSR)
parameter sweeps, and single/batch simulation case management.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from src.pyvawt.multiple.aerodynamics import NeuralFoilAirfoilWrapper, readaerodyn
from src.pyvawt.multiple.export import DEFAULT_RESULTS_DIR, export_coupled_case_results
from src.pyvawt.multiple.settings import load_config
from src.pyvawt.multiple.simulation import Environment, Turbine, actuatorcylinder
from src.pyvawt.ui.ui import UI, MultiTurbineUI

logger = logging.getLogger(__name__)

# Default operational boundaries for TSR sweeps
DEFAULT_TSR_START: float = 1.0
DEFAULT_TSR_END: float = 7.0
DEFAULT_TSR_POINTS: int = 20


class SimulationContext(NamedTuple):
    """
    Data container holding initialized structures and parameters for a simulation run.

    Attributes
    ----------
    turbines : list of Turbine
        List of initialized `Turbine` instances defining the array layout.
    env : Environment
        Initialized physical environment (fluid medium and free-stream velocity).
    simulation_params : dict of {str : Any}
        Solver numerical and configuration parameters.
    turbine_params : dict of {str : Any}
        Rotor geometrical and operational parameters.
    environment_params : dict of {str : Any}
        Physical properties of the fluid domain ($\rho$, $\mu$).
    radius : float
        Turbine rotor radius in meters.
    ntheta : int
        Number of azimuthal discretization points along the rotor orbit.
    """

    turbines: list[Turbine]
    env: Environment
    simulation_params: dict[str, Any]
    turbine_params: dict[str, Any]
    environment_params: dict[str, Any]
    radius: float
    ntheta: int


def initialize_turbine_and_environment(config: dict[str, Any]) -> SimulationContext:
    """
    Instantiate turbine geometry, fluid domain, and aerodynamic polars.

    Constructs the physical domain based on the provided configuration dictionary,
    binding airfoil performance lookup wrappers (NeuralFoil LUTs or classic AeroDyn).

    Parameters
    ----------
    config : dict of {str : Any}
        Validated dictionary containing simulation parameters.

    Returns
    -------
    SimulationContext
        Data container populated with instantiated domain objects.

    Raises
    ------
    KeyError
        If mandatory keys or classic airfoil file paths are missing.
    """
    turbine_params = config["turbine"]
    environment_params = config["environment"]
    solver_params = config.get("solver", {})

    # Extract geometrical and operational parameters
    r = float(turbine_params["r"])
    twist = float(turbine_params["twist"])
    delta = float(turbine_params["delta"])
    b_blades = int(turbine_params["B"])
    center_x = turbine_params["centerX"]
    center_y = turbine_params["centerY"]
    omega = float(turbine_params["Omega"])
    ntheta = int(turbine_params["ntheta"])

    # Extract fluid medium properties
    vinf_raw = environment_params["Vinf"]
    vinf = float(vinf_raw[0]) if isinstance(vinf_raw, list) else float(vinf_raw)
    rho = float(environment_params["rho"])
    mu = float(environment_params["mu"])

    num_turbines = int(solver_params.get("num_turbines", 1))

    # Normalize chord vector across all turbine instances
    raw_chord = turbine_params["chord"]
    chord_list = [float(c) for c in raw_chord] if isinstance(raw_chord, list) else [float(raw_chord)] * num_turbines
    if len(chord_list) < num_turbines:
        chord_list.extend([chord_list[-1]] * (num_turbines - len(chord_list)))

    # Instantiate aerodynamic evaluation backends
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
            raise KeyError("Airfoil file path missing under configuration key 'solver.file.path'.")
        classic_profile = readaerodyn(aero_profile)
        turbines_airfoils = [classic_profile] * num_turbines

    # Calculate turbine center coordinates (default: 4R streamwise offset)
    cx_list = [float(center_x) + i * 4.0 * r for i in range(num_turbines)] if isinstance(center_x, (int, float)) else [float(x) for x in center_x]
    cy_list = [float(center_y) for _ in range(num_turbines)] if isinstance(center_y, (int, float)) else [float(y) for y in center_y]

    if len(cx_list) < num_turbines:
        cx_list.extend([cx_list[-1] + 4.0 * r for _ in range(num_turbines - len(cx_list))])
    if len(cy_list) < num_turbines:
        cy_list.extend([cy_list[-1] for _ in range(num_turbines - len(cy_list))])

    # Instantiate physical object models
    turbines = [
        Turbine(r, chord_list[i], twist, delta, b_blades, turbines_airfoils[i], omega, cx_list[i], cy_list[i])
        for i in range(num_turbines)
    ]

    env = Environment(vinf, rho, mu)

    return SimulationContext(
        turbines=turbines,
        env=env,
        simulation_params=solver_params,
        turbine_params=turbine_params,
        environment_params=environment_params,
        radius=r,
        ntheta=ntheta
    )


def _execute_tsr_sweep(
    turbines: list[Turbine],
    env: Environment,
    ntheta: int,
    tsr_vec: np.ndarray,
    var_omega_vinf: int,
    vinf: float,
    radius: float,
    num_turbines: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Execute a Tip Speed Ratio (TSR) parameter sweep over the operational range.

    Parameters
    ----------
    turbines : list of Turbine
        List of initialized `Turbine` instances.
    env : Environment
        Fluid medium environment object.
    ntheta : int
        Azimuthal resolution (number of angular steps along blade orbit).
    tsr_vec : np.ndarray
        Array of Tip Speed Ratio ($\lambda$) values to sweep.
    var_omega_vinf : int
        Strategy flag: 0 to vary rotational speed $\Omega$; 1 to vary free-stream speed $V_{\infty}$.
    vinf : float
        Nominal wind velocity in m/s.
    radius : float
        Rotor radius in meters.
    num_turbines : int
        Total number of turbines in the coupled setup.

    Returns
    -------
    cp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Power coefficient ($C_p$) results.
    ct_vec : np.ndarray, shape (N_tsr, N_turbines)
        Thrust coefficient ($C_t$) results.
    rp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Radial force coefficient outputs.
    tp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Tangential force coefficient outputs.
    zp_vec : np.ndarray, shape (N_tsr, N_turbines)
        Axial force coefficient outputs.
    theta_vec : np.ndarray, shape (N_tsr, ntheta)
        Azimuthal angle discretization points.
    warnings : list of str
        List of solver warning messages collected during the sweep.

    Raises
    ------
    ValueError
        If an invalid `var_omega_vinf` strategy flag is supplied.
    """
    n_points = len(tsr_vec)
    cp_vec = np.zeros((n_points, num_turbines))
    ct_vec = np.zeros((n_points, num_turbines))
    rp_vec = np.zeros((n_points, num_turbines))
    tp_vec = np.zeros((n_points, num_turbines))
    zp_vec = np.zeros((n_points, num_turbines))
    theta_vec = np.zeros((n_points, ntheta))

    w_guess = None
    warnings: list[str] = []

    UI.section("SWEEP EXECUTION")
    sweep_start_time = time.perf_counter()

    for i, tsr in enumerate(tsr_vec):
        # Adjust operational point variables
        if var_omega_vinf == 0:
            for turbine in turbines:
                turbine.Omega = vinf * tsr / radius
        elif var_omega_vinf == 1:
            for turbine in turbines:
                turbine.Omega = 13.62 * 2 * np.pi / 60.0
            env.Vinf = turbines[0].Omega * radius / tsr
        else:
            raise ValueError(f"Invalid omega/vinf strategy configuration: {var_omega_vinf}")

        # Execute Actuator Cylinder numerical engine
        res = actuatorcylinder(turbines, env, ntheta, w0=w_guess)

        # Dynamic slicing for safe unpacking of variable solver returns
        ct, cp, rp, tp, zp, theta = res[:6]
        w_guess = res[6] if len(res) > 6 else None

        # Collect optional warnings from solver
        if len(res) > 7:
            for extra in res[7:]:
                if isinstance(extra, list):
                    warnings.extend(extra)
                elif isinstance(extra, str):
                    warnings.append(extra)

        # Map results per turbine instance
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

        # Report progress
        elapsed = time.perf_counter() - sweep_start_time
        MultiTurbineUI.print_progress(i + 1, n_points, elapsed)

    return cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec, warnings


def run_simulation_case(params: tuple[int, int, float, float, float]) -> dict[str, Any]:
    """
    Run an end-to-end simulation case (setup, execution, export, and UI reporting).

    Parameters
    ----------
    params : tuple of (int, int, float, float, float)
        Tuple containing case metadata: `(case_idx, total_cases, chord, solidity, vinf)`.

    Returns
    -------
    dict of {str : Any}
        Execution log summary containing metadata and runtime metrics.

    Raises
    ------
    KeyError
        If required classic airfoil configurations are missing.
    Exception
        Re-raises any uncaught runtime exception after logging.
    """
    case_idx, total_cases, chord, solidity, vinf = params
    config = load_config()

    solver_cfg = config.get("solver", {})
    method = solver_cfg.get("method", "neuralfoil")
    use_neuralfoil = (method == "neuralfoil")

    # Determine airfoil identifier for output naming
    if use_neuralfoil:
        nf_cfg = solver_cfg.get("neuralfoil", {})
        raw_airfoil = nf_cfg.get("airfoil", ["naca0018"])
        airfoil_name = raw_airfoil[0] if isinstance(raw_airfoil, list) else str(raw_airfoil)
    else:
        file_cfg = solver_cfg.get("file", {})
        airfoil_file = file_cfg.get("path")
        if not airfoil_file:
            raise KeyError("Classic profile path missing in configuration key 'solver.file.path'.")
        airfoil_name = Path(airfoil_file).stem

    num_turbines = int(solver_cfg.get("num_turbines", 1))
    blades = int(config.get("turbine", {}).get("B", 3))
    radius = round(chord * blades / solidity, 4)

    fixed_param = str(solver_cfg.get("fixed_parameter", "vinf")).lower()
    var_omega_vinf = 0 if fixed_param == "vinf" else 1

    # Override target configuration values for current case
    config["turbine"]["chord"] = [chord] * num_turbines
    config["turbine"]["solidity"] = [solidity] * num_turbines
    config["environment"]["Vinf"] = [vinf] if isinstance(config["environment"]["Vinf"], list) else vinf
    config["turbine"]["r"] = radius

    case_name = f"{airfoil_name}_turb{num_turbines}_b{blades}_r{radius}_ch{chord}_sol{solidity}_vinf{vinf}".replace(".", "p")

    # 1. Initialize simulation domain
    t_start_setup = time.perf_counter()
    context = initialize_turbine_and_environment(config)
    jit_setup_time = time.perf_counter() - t_start_setup

    MultiTurbineUI.print_init(turbines=context.turbines, jit_time=jit_setup_time, mode_coupled=True)

    start_time = time.perf_counter()

    try:
        tsr_vec = np.linspace(DEFAULT_TSR_START, DEFAULT_TSR_END, DEFAULT_TSR_POINTS)

        # 2. Run TSR sweep
        cp_vec, ct_vec, rp_vec, tp_vec, zp_vec, theta_vec, warnings = _execute_tsr_sweep(
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

        # 3. Export results
        case_dir = export_coupled_case_results(
            case_name=case_name,
            config=config,
            turbines=context.turbines,
            tsr_vec=tsr_vec,
            cp_vec=cp_vec,
            ct_vec=ct_vec,
            rp_vec=rp_vec,
            tp_vec=tp_vec,
            zp_vec=zp_vec,
            base_results_dir=DEFAULT_RESULTS_DIR
        )

        output_dir_str = str(case_dir.resolve()) if case_dir else "Disabled (save=false)"

        # 4. Report via UI
        MultiTurbineUI.print_results(
            turbines=context.turbines,
            total_time=elapsed,
            output_dir=output_dir_str,
            cp_results=cp_vec.T,
            ct_results=ct_vec.T,
            tsr_vec=tsr_vec,
            warnings=warnings
        )

        return {
            "name": case_name,
            "airfoil": airfoil_name,
            "num_turbines": num_turbines,
            "blades": blades,
            "radius": radius,
            "chord": chord,
            "solidity": solidity,
            "vinf": vinf,
            "status": "OK",
            "time_sec": round(elapsed, 2)
        }

    except Exception as err:
        logger.error(f"Execution failed for case '{case_name}': {err}", exc_info=True)
        raise

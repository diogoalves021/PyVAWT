# PyVAWT: Vertical Axis Wind Turbine Aerodynamic Simulator

**PyVAWT** is a Python package developed for the aerodynamic simulation of Vertical Axis Wind Turbines (VAWTs) based on the **Actuator Cylinder** theory. The program incorporates high-fidelity aerodynamic loss models while maintaining low computational cost, allowing simulations to complete in just a few minutes on standard desktop computers.

---

## Key Features

* **Fast Aerodynamic Modeling:** Actuator Cylinder solver integrated with **NeuralFoil** for rapid airfoil polar predictions via JIT compilation and neural networks.
* **Flexible Modules:** Support for **single turbine** simulations (with 2D/3D options and advanced aerodynamic submodels) and **multiple coupled turbines** (2D aerodynamic interaction).
* **Automated Parametric Sweep:** Automated execution of parameter combinations (chord, solidity, wind speed, and airfoil profiles).
* **Physical Submodels:** Support for tip loss (*Tip Loss*), dynamic stall (*Dynamic Stall*), and flow curvature (*Flow Curvature*) corrections.
* **Comprehensive Export Options:** Generation of data tables (`.dat`, `.csv`), performance curves ($C_P \times \text{TSR}$, $C_T \times \text{TSR}$), and azimuthal distributions ($C_P(\theta)$).

> **Important Note on Parametric Analysis:** The automated parametric sweep functionality (providing array inputs for `chord`, `solidity`, `Vinf`, etc.) **is exclusively available for single-turbine simulations in 2D mode** (`simulation3d.enabled: false`). The 3D and Multiple Turbines modules operate strictly with scalar inputs per run.

---

## Project Origin & Funding

This software was developed as part of an academic research project focused on the aerodynamic modeling and optimization of vertical axis wind turbines.

This work was supported by the **São Paulo Research Foundation (FAPESP)**.

---

## Requirements & Installation

### Minimum Requirements
* **Operating System:** Linux, macOS, or Windows.
* **Python:** Version 3.13 or higher.
* **Environment Manager:** `uv` (recommended for fast dependency management).

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/pyvawt.git](https://github.com/your-username/pyvawt.git)
   cd pyvawt
   ```

2. **Create the virtual environment and install dependencies via `uv`:**
   ```bash
   uv sync
   ```

---

## Configuration Files (`YAML`)

The program uses hierarchical configuration files to define simulation parameters.

### 1. Single Turbine (`config.yaml`)

```yaml
turbine:
  r: 1.0          # Rotor radius [m]
  H: 2.0          # Rotor height [m]
  twist: 0.0      # Twist angle [deg]
  delta: 0.0      # Pitch angle [deg]
  chord: 0.15     # Blade chord [m] (or array for 2D sweeps)
  B: 2            # Number of blades
  solidity: 0.3   # Solidity [-] (or array for 2D sweeps)
  centerX: 0.0
  centerY: 0.0
  Omega: 15.0     # Angular velocity [rad/s]
  ntheta: 36

environment:
  Vinf: 10.0      # Free-stream wind speed [m/s]
  rho: 1.225      # Air density [kg/m³]
  mu: 1.789e-5    # Dynamic viscosity [Pa·s]

solver:
  fixed_parameter: "vinf" # "vinf" or "omega"
  tsr:
    min: 1.0
    max: 6.0
    n_points: 20
  method: "neuralfoil"
  neuralfoil:
    airfoil: ["NACA0012"]
    model_size: "medium"
  simulation3d:
    enabled: false # Requires 'false' for parametric sweeps

submodels:
  tip_loss: true
  dynamic_stall: false
  flow_curvature:
    enabled: true
    normalized_hook_point: 0.5

output:
  save: true
  save_config: true
  save_plot: true
  data_file:
    format: "dat" # "dat" or "csv"
    include_header: true
  plot_image:
    format: "png" # "png" or "eps"
    dpi: 300
```

### 2. Multiple Turbines (`config_multiple.yaml`)

For multiple turbine arrays, specify the number of rotors and their spatial coordinates in array format:

```yaml
turbine:
  centerX: [0.0, 3.0] # X positions of turbines [m]
  centerY: [0.0, 0.0] # Y positions of turbines [m]

solver:
  num_turbines: 2
```

---

## Running Simulations

Simulations are executed from the root directory using `uv`:

### Single Turbine Simulation
```bash
uv run python3 -m src.pyvawt.single.main
```

To inspect all loaded configuration parameters prior to execution:
```bash
uv run python3 -m src.pyvawt.single.main --show-config
```

### Multiple Turbines Simulation
```bash
uv run python3 -m src.pyvawt.multiple.main
```

---

## Output Structure (Results)

Upon completion, a timestamped directory is created inside the results folder containing:

```text
results/
└── YYYYMMDD_HHMMSS_run/
    ├── config_used.yaml          # Copy of the configuration file used
    ├── results_NACA0012.dat      # Numerical results (TSR, Cp, Ct)
    ├── cp_curve_NACA0012.png     # Cp vs TSR plot
    └── cp_theta_NACA0012.png     # Azimuthal distribution Cp(theta) plot
```

---

## Repository Structure

```text
pyvawt/
├── config/
│   ├── config.yaml              # Default configuration (Single Turbine)
│   └── config_multiple.yaml     # Default configuration (Multiple Turbines)
├── pyproject.toml           # Project configuration & dependencies
├── uv.lock                  # uv lockfile
├── README.md                # Main documentation
└── src/
    └── pyvawt/
        ├── single/          # Single turbine module (2D/3D/Parametric)
        │   └── main.py
        └── multiple/        # Multiple turbine array module (Only 2D)
            └── main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

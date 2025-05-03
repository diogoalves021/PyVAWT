# PyVAWT

PyVAWT is a Python program for simulating Vertical Axis Wind Turbines (VAWTs) using the actuator disk method. This method models the aerodynamic behavior of VAWTs by representing the rotor as a series of cylinders interacting with the wind flow.

## Description

The goal of the project is to provide insights into the performance of wind turbines, making it ideal for researchers and engineers involved in the optimization and performance analysis of turbines.

**Current status:**  
The project is under active development. Basic features have already been implemented, including:

1. Simulation of a single turbine
2. Simulation of two turbines
3. Simple wake model
4. Reading simulation parameters from a `.json` file

> **Note:** This project is still in its early stages of development. It is not recommended for critical engineering analysis, as accuracy and features are subject to significant changes.

---

## Features

### 1. Aerodynamic Data Generation (`data_generation/generator.py`)

Generates lift (Cl) and drag (Cd) curves using the `neuralfoil` module from the [AeroSandbox](https://aerosandbox.com/) library, based on the provided simulation conditions.

- Choose any NACA airfoil
- Define Reynolds and Mach numbers
- Ability to simulate multiple airfoils automatically

**✓ High precision**  
**✗ High generation time (~10 minutes per airfoil)**

Ideal for detailed and reliable aerodynamic performance analysis.

---

### 2. Reading Pre-Generated Data (`data_reading/reader.py`)

Imports files with Cl/Cd curves obtained from other tools (such as XFoil, QBlade, or AeroSandbox itself) and interpolates the data for use in the simulation.

- Supports multiple input formats
- Fast simulation (typically ~5 seconds)

**✓ High simulation speed**  
**✗ Lower precision (dependent on the quality of input data)**

Ideal for quick testing, parametric studies, or optimizations.

---

## Usage

Create a configuration file `config.json` with the simulation parameters:


```json
{
    "turbine": {
        "r": 17.5,
        "twist": 0.0,
        "delta": 0.0,
        "chord": 1.75,
        "B": 2,
        "solidity": 0.1,
        "centerX": 0,
        "centerY": 0,
        "Omega": 0.0,
        "ntheta": 36
    },
    "environment": {
        "Vinf": 1.0,
        "rho": 1.225,
        "mu": 1.7894e-5
    },
    "simulation": {
        "var_omega_vinf": 0,
        "num_turbines": 2,
        "aero_profile": "data/NACA_0012_mod.dat"
    }
}


After setting up your environment and configuring the simulation parameters, you can run the code using the following command:

```bash
python3 -m examples.Solidity_chord_vinf_Fallstudie
```

This will start the simulation, and the results will be displayed according to the parameters specified in the JSON configuration file, and in this case, according to the aerodynamic profiles you select in the code. The results will be saved in folders with the name of the selected parameters. In these folders there will be a json file with the parameters used in the simulation, an .eps file with the data graph and a .dat file with the data.

## Directories Structure

```

project_root/
│
├── src/                       # Main application code
│   ├── __init__.py
│   ├── main.py                # Main entry point
│   ├── config/                # Static configuration files
│   │   └── config.json
│   ├── data_generation/       # Cl/Cd data generation
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── data_reading/          # Cl/Cd data reading and interpolation
│   │   ├── __init__.py
│   │   └── reader.py
│   ├── simulation/            # Simulation core logic
│   │   ├── __init__.py
│   │   └── simulator.py
│   └── utils/                 # Reusable utility functions
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                     # Unit and functional tests
│   ├── __init__.py
│   ├── test_generator.py
│   ├── test_reader.py
│   └── test_simulation.py
│
├── examples/                  # Usage examples and case studies
│   └── example_case_1/
│       ├── params.json
│       └── run_case.py
│
├── results/                   # Simulation results (organized by case/date)
│   └── case_01/
│       └── results.json
│
├── data/                      # Raw data or airfoil definitions
│   └── airfoil_001.dat
│
├── README.md
├── requirements.txt
└── setup.py                   # Package installation file

```

## Dependecies

The dependencies for this project are listed in the `pyproject.toml` file. To install the required dependencies, you can use [Poetry](https://python-poetry.org/) or [pip](https://pip.pypa.io/en/stable/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# PyVAWT
PyVAWT is a Python program for simulating vertical-axis wind turbines (VAWTs) using the actuator cylinder method. This method models the aerodynamic behavior of VAWTs by representing the rotor as a series of cylinders that interact with the wind flow.

## Description

The project aims to provide insights into turbine performance and is ideal for researchers and engineers working on wind turbine optimization and performance analysis.

**Current Status:**  
The project is in active development. Currently, basic functionality has been implemented, but improvements are being made continuously, such as:

1. One turbine simulation
2. Two turbines simulation
3. Simple wake model
4. Simulation paramters from a .json file

**Warning:**
As mentioned above, this project is in early development. Therefore, it is not recommended for serious simulation purposes or even for initial analysis. The functionality and accuracy may be incomplete or subject to significant changes as development progresses.

## Usage

Here’s an example of how to simulate a vertical-axis wind turbine using a JSON configuration file for the simulation parameters:

**Create a JSON file** (e.g., `config.json`) that includes the simulation parameters:

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
python src/main.py
```

This will start the simulation, and the results will be displayed according to the parameters specified in the JSON configuration file and saved in the HDF5 and .dat files.

## Dependecies

The dependencies for this project are listed in the `pyproject.toml` file. To install the required dependencies, you can use [Poetry](https://python-poetry.org/) or [pip](https://pip.pypa.io/en/stable/).
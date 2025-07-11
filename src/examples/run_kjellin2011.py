from src.pyvawt.main import run_simulation, load_config, save_config
import os

# Load the original config file
config = load_config('src/pyvawt/config/config.json')

# Changing the parameters
config['turbine']['r'] = 3.0
config['turbine']['twist'] = 0.0 
config['turbine']['delta'] = 0.0 
config['turbine']['chord'] = [0.25]
config['turbine']['B'] = 3 
config['turbine']['solidity'] = [0.125]
config['turbine']['centerX'] = 0
config['turbine']['centerY'] = 0 
config['turbine']['Omega'] = 5.03 
config['turbine']['ntheta'] = 36

config['environment']['Vinf'] = [5.0]
config['environment']['rho'] = 1.225 
config['environment']['mu'] = 1.7894e-05 

config['simulation']['var_omega_vinf'] = 1 
config['simulation']['num_turbines'] = 1 
config['simulation']['airfoil'] = ["naca0021"]

# Saving temporary config file
temp_path = 'src/pyvawt/config/config.json'
save_config(config, temp_path)

# Run simulation
run_simulation()
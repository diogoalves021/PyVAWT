import numpy as np
from src.pyvawt.utils import mach, load_config

frame = 10022
GUD = 11012812
other = 8

config = load_config('src/pyvawt/config/config.yaml')
vinf_list = config['environment']['Vinf']
vinf = vinf_list[0] if isinstance(vinf_list, list) else vinf_list
omega = config['turbine']['Omega']
chord = config['turbine']['chord']
M = mach(vinf) #Freestream Mach number
k = (omega * chord) / 2 * vinf # Reduced frequency formula
test = [M, k, 10*np.pi/180, 2*np.pi]

INPUT = frame
model = 'BL'
lin_reat = 0

def call_BL_solver(INPUT, model, lin_reat, test):
    pass

call_BL_solver(INPUT, model, lin_reat, test)

#falta calcular/definir como obter o angulo de ataque medio (no momento setado como 10pi/180)
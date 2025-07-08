import json
import unittest
import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.pyvawt import readaerodyn, actuatorcylinder, Turbine, Environment

atol = 1e-6

def load_config():
    with open('config/config.json', 'r') as f:
        return json.load(f)

def initialize_turbine_and_environment(config):
    turbine_params = config['turbine']
    environment_params = config['environment']
    simulation_params = config['simulation']

    #Turbine parameters
    r = turbine_params['r']
    twist = turbine_params['twist']
    delta = turbine_params['delta']
    af = readaerodyn(simulation_params['aero_profile'])
    chord = turbine_params['chord']
    B = turbine_params['B']
    solidity = turbine_params['solidity']
    centerX = turbine_params['centerX']
    centerY = turbine_params['centerY']
    Omega = turbine_params['Omega']
    ntheta = turbine_params['ntheta']

    #Environment parameters
    Vinf = environment_params['Vinf']
    rho = environment_params['rho']
    mu = environment_params['mu']

    #Creating turbine
    #turbines = [Turbine(r, chord, twist, delta, B, af, Omega, centerX, centerY)]
    turbines = [None] * 1  # Cria uma lista de tamanho 1 vazia (com um elemento `None`)
    turbines[0] = Turbine(r, chord, twist, delta, B, af, Omega, 0.0, 0.0)  # Inicializa a turbina

    #Creating env
    env = Environment(Vinf, rho, mu)

    return turbines, env, simulation_params, turbine_params, environment_params, r, ntheta


def run_simulation(turbines, env, simulation_params, r, ntheta, Vinf, num_turbines, turbine_params):
    var_omega_vinf = simulation_params['var_omega_vinf']

    #------------------------------
    #One Turbine
    #------------------------------
    if num_turbines == 1:
        n = 20
        tsrvec = np.linspace(1, 7, n)
        CPvec = np.zeros(n)
        CTvec = np.zeros(n)
        Rpvec = np.zeros(n)
        Tpvec = np.zeros(n)
        Zpvec = np.zeros(n)
        thetavec = np.zeros((n, ntheta))

        if var_omega_vinf == 0:
            for i, tsr in enumerate(tsrvec):
                turbines[0].Omega = Vinf * tsr / r
                CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)
                CPvec[i] = CP[0]
                CTvec[i] = CT[0]
                Rpvec[i] = Rp[0].item()
                Tpvec[i] = Tp[0].item()
                Zpvec[i] = Zp[0].item()
                thetavec[i, :] = theta

        elif var_omega_vinf == 1:
            for i, tsr in enumerate(tsrvec):
                turbines[0].Omega = 13.62 * 2 * np.pi / 60.0
                env.Vinf = turbines[0].Omega * r / tsr
                CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)
                CPvec[i] = CP[0]
                CTvec[i] = CT[0]
                Rpvec[i] = Rp[0].item()
                Tpvec[i] = Tp[0].item()
                Zpvec[i] = Zp[0].item()
                thetavec[i, :] = theta

        else:
            print('ERRO! É preciso digitar 0 ou 1 para no parâmetro de entrada para variar Omega ou Vinf.')

        print('Simulation completed.')

        #Salving results as a HDF5 file
        #with h5py.File('test/single_unit_test_data.h5', 'w') as f:
        #    f.create_dataset('CPvec_old', data=CPvec)
        #    f.create_dataset('CTvec_old', data=CTvec)
        #    f.create_dataset('Rpvec_old', data=Rpvec)
        #    f.create_dataset('Tpvec_old', data=Tpvec)
        #    f.create_dataset('Zpvec_old', data=Zpvec)
        #    f.create_dataset('thetavec_old', data=thetavec)

        #Carregando os resultados
        with h5py.File('test/single_unit_test_data.h5', 'r') as f:
            CPvec_old = np.array(f["CPvec_old"])
            CTvec_old = np.array(f["CTvec_old"])
            Rpvec_old = np.array(f["Rpvec_old"])
            Tpvec_old = np.array(f["Tpvec_old"])
            Zpvec_old = np.array(f["Zpvec_old"])
            thetavec_old = np.array(f["thetavec_old"])

        print("CP atual:", CPvec)
        print("CP esperado:", CPvec_old)
        print("Erro relativo (%):", np.abs(((CPvec_old - CPvec)/CPvec_old)*100))
        print("Tolerância:", atol)

        #Teste de comparação
        #assert np.allclose(CPvec, CPvec_old, atol=atol)
        #assert np.allclose(CTvec, CTvec_old, atol=atol)
        #assert np.allclose(Rpvec, Rpvec_old, atol=atol)
        #assert np.allclose(Tpvec, Tpvec_old, atol=atol)
        #assert np.allclose(Zpvec, Zpvec_old, atol=atol)
        #assert np.allclose(thetavec, thetavec_old, atol=atol)

        #Saving results as a .dat file
        data_to_save = np.column_stack((tsrvec, CPvec, CTvec, Rpvec, Tpvec, Zpvec))
        header = "TSR CP CT Rp Tp Zp"
        np.savetxt("results/1turbine_results_korrekturfaktor.dat", data_to_save, header=header, fmt="%.6f", delimiter="\t")

        print('\nGenerating results...')

        #Ploting results
        plt.figure(figsize=(10, 5))
        plt.plot(tsrvec[tsrvec <=6.5], CPvec[tsrvec <=6.5], color='blue', label='$C_p$')
        plt.title('Gráfico de $C_p$ x TSR ($\\lambda$)')
        plt.grid()
        plt.legend()
        plt.show()

    #------------------------------
    #Two Turbines
    #------------------------------

    elif num_turbines == 2:
        tsr = 3.5
        Omega = Vinf * tsr / r

        #Creating turbines
        turbines = [
            Turbine(r, turbine_params['chord'], turbine_params['twist'], turbine_params['delta'], turbine_params['B'], readaerodyn(simulation_params['aero_profile']), Omega, 0.0, 0.0),
            Turbine(r, turbine_params['chord'], turbine_params['twist'], turbine_params['delta'], turbine_params['B'], readaerodyn(simulation_params['aero_profile']), -Omega, 0.0, 2 * r)
        ]

        #Calling actuator cylinder function
        CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)

        print('Simulation completed.')

        #Saving results as a HDF5 file
        #with h5py.File('test/dual_unit_test_data.h5', 'w') as file:
        #    file.create_dataset('CP_old', data=CP)
        #    file.create_dataset('CT_old', data=CT)
        #    file.create_dataset('Rp_old', data=Rp)
        #    file.create_dataset('Tp_old', data=Tp)
        #    file.create_dataset('Zp_old', data=Zp)
        #    file.create_dataset('theta_old', data=theta)

        #Reading the HDF5 file data
        with h5py.File('test/dual_unit_test_data.h5', 'r') as file:
            CP_old = file['CP_old'][:]
            CT_old = file['CT_old'][:]
            Rp_old = file['Rp_old'][:]
            Tp_old = file['Tp_old'][:]
            Zp_old = file['Zp_old'][:]
            theta_old = file['theta_old'][:]

        print("CP atual:", CP)
        print("CP esperado:", CP_old)
        print("Diferença absoluta:", np.abs(CP - CP_old))
        print("Tolerância:", atol)

        #Testing (making sure the values are approximately equal)
        assert np.allclose(CP, CP_old, atol=atol)
        assert np.allclose(CT, CT_old, atol=atol)
        assert np.allclose(Rp, Rp_old, atol=atol)
        assert np.allclose(Tp, Tp_old, atol=atol)
        assert np.allclose(Zp, Zp_old, atol=atol)
        assert np.allclose(theta, theta_old, atol=atol)

        print('\nGenerating results...')

        #Ploting results
        plt.figure()
        plt.plot(theta, r * Tp[:, 0], label='Turbina 1') 
        plt.plot(theta, r * Tp[:, 1], label='Turbina 2')  
        plt.xlabel(r'$\theta$')
        plt.ylabel('Q (torque)')
        plt.xlim([0, 2 * np.pi])
        plt.legend()
        plt.show()

        #Saving results as a .dat file
        dados = np.column_stack((theta, r * Tp[:, 0], r * Tp[:, 1])) 
        np.savetxt('results/two_turbines_results_python.dat', dados, header="theta, torque_1, torque_2", comments='')

config = load_config()
turbines, env, simulation_params, turbine_params, environment_params, r, ntheta = initialize_turbine_and_environment(config)
initialize_turbine_and_environment(config)
num_turbines = simulation_params['num_turbines']
run_simulation(turbines, env, simulation_params, r, ntheta, environment_params['Vinf'], num_turbines, turbine_params)
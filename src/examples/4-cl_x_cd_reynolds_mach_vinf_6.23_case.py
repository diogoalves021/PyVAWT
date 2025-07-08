import json
import unittest
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from src.pyvawt import readaerodyn, actuatorcylinder, Turbine, Environment

atol = 1e-6

def load_config(novo_perfil=None):
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    if novo_perfil:
        config['simulation']['aero_profile'] = novo_perfil
    return config

def initialize_turbine_and_environment(config):
    turbine_params = config['turbine']
    environment_params = config['environment']
    simulation_params = config['simulation']

    # Parâmetros da turbina
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

    # Parâmetros do ambiente
    Vinf = environment_params['Vinf']
    rho = environment_params['rho']
    mu = environment_params['mu']

    # Criação da turbina
    turbines = [None] * 1
    turbines[0] = Turbine(r, chord, twist, delta, B, af, Omega, 0.0, 0.0)

    # Criação do ambiente
    env = Environment(Vinf, rho, mu)

    return turbines, env, simulation_params, turbine_params, environment_params, r, ntheta

def run_simulation(turbines, env, simulation_params, r, ntheta, Vinf, num_turbines, turbine_params):
    var_omega_vinf = simulation_params['var_omega_vinf']
    Omega = turbine_params['Omega']
    # Usaremos o nome base do arquivo do perfil para nomear os arquivos de saída
    profile_name = os.path.basename(simulation_params['aero_profile']).replace('.txt', '')
    
    # ------------------------------
    # Simulação para uma turbina
    # ------------------------------
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
                #turbines[0].Omega = 13.62 * 2 * np.pi / 60.0
                turbines[0].Omega = Omega
                env.Vinf = turbines[0].Omega * r / tsr
                CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)
                CPvec[i] = CP[0]
                CTvec[i] = CT[0]
                Rpvec[i] = Rp[0].item()
                Tpvec[i] = Tp[0].item()
                Zpvec[i] = Zp[0].item()
                thetavec[i, :] = theta

        else:
            print('ERRO! É preciso digitar 0 ou 1 para o parâmetro var_omega_vinf.')

        print('Simulation completed for profile:', profile_name)

        # Se desejar realizar validações com dados de teste salvos previamente via HDF5, descomente os trechos abaixo:
        """
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

        assert np.allclose(CPvec, CPvec_old, atol=atol)
        assert np.allclose(CTvec, CTvec_old, atol=atol)
        assert np.allclose(Rpvec, Rpvec_old, atol=atol)
        assert np.allclose(Tpvec, Tpvec_old, atol=atol)
        assert np.allclose(Zpvec, Zpvec_old, atol=atol)
        assert np.allclose(thetavec, thetavec_old, atol=atol)
        """

        # Salvando resultados como um arquivo .dat com o nome baseado no perfil utilizado
        data_to_save = np.column_stack((tsrvec, CPvec, CTvec, Rpvec, Tpvec, Zpvec))
        header = "TSR\tCP\tCT\tRp\tTp\tZp"
        out_filename = f"results/1turbine_results_{profile_name}.dat"
        np.savetxt(out_filename, data_to_save, header=header, fmt="%.6f", delimiter="\t")
        print(f"Arquivo de resultados salvo: {out_filename}")

        # Gerando gráfico dos resultados
        plt.figure(figsize=(10, 5))
        plt.plot(tsrvec[tsrvec >= 0], CPvec[tsrvec >= 0], color='blue', label='$C_p$')
        plt.title(f'Gráfico de $C_p$ x TSR ($\\lambda$) para {profile_name}')
        plt.legend()
        plt.show()

    # ------------------------------
    # Simulação para duas turbinas
    # ------------------------------
    elif num_turbines == 2:
        tsr = 3.5
        Omega = Vinf * tsr / r

        # Criação de duas turbinas com perfil modificado
        turbines = [
            Turbine(r, turbine_params['chord'], turbine_params['twist'], turbine_params['delta'],
                    turbine_params['B'], readaerodyn(simulation_params['aero_profile']), Omega, 0.0, 0.0),
            Turbine(r, turbine_params['chord'], turbine_params['twist'], turbine_params['delta'],
                    turbine_params['B'], readaerodyn(simulation_params['aero_profile']), -Omega, 0.0, 2 * r)
        ]

        # Chamada da função actuatorcylinder
        CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)
        print('Simulation completed for profile:', profile_name)

        # Se desejar utilizar validação via HDF5, descomente o bloco abaixo:
        """
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

        assert np.allclose(CP, CP_old, atol=atol)
        assert np.allclose(CT, CT_old, atol=atol)
        assert np.allclose(Rp, Rp_old, atol=atol)
        assert np.allclose(Tp, Tp_old, atol=atol)
        assert np.allclose(Zp, Zp_old, atol=atol)
        assert np.allclose(theta, theta_old, atol=atol)
        """

        # Gerando gráfico de torque para as duas turbinas
        plt.figure()
        plt.plot(theta, r * Tp[:, 0], label='Turbina 1')
        plt.plot(theta, r * Tp[:, 1], label='Turbina 2')
        plt.xlabel(r'$\theta$')
        plt.ylabel('Q (torque)')
        plt.xlim([0, 2 * np.pi])
        plt.legend()
        plt.title(f'Resultados para duas turbinas - {profile_name}')
        plt.show()

        # Salvando resultados em um arquivo .dat
        dados = np.column_stack((theta, r * Tp[:, 0], r * Tp[:, 1]))
        out_filename = f"results/two_turbines_results_{profile_name}.dat"
        np.savetxt(out_filename, dados, header="theta, torque_1, torque_2", comments='')
        print(f"Arquivo de resultados salvo: {out_filename}")
    else:
        print("Número de turbinas inválido.")

def runtest():
    # Lista dos caminhos dos perfis aerodinâmicos a serem testados
    perfis = [
        'data/neuralfoil/naca0018_cl_cd_Re=500000.0_M=0.0.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=500000.0_M=0.1.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=500000.0_M=0.2.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=500000.0_M=0.3.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=1000000.0_M=0.0.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=1000000.0_M=0.1.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=1000000.0_M=0.2.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=1000000.0_M=0.3.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=5000000.0_M=0.0.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=5000000.0_M=0.1.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=5000000.0_M=0.2.txt',
        'data/neuralfoil/naca0018_cl_cd_Re=5000000.0_M=0.3.txt'
    ]

    # Garante que a pasta de resultados exista
    os.makedirs("results", exist_ok=True)
    
    for perfil in perfis:
        print(f"\nExecutando simulação com o perfil: {perfil}")
        config = load_config(novo_perfil=perfil)
        turbines, env, simulation_params, turbine_params, environment_params, r, ntheta = initialize_turbine_and_environment(config)
        num_turbines = simulation_params['num_turbines']
        # Cada simulação é executada e os resultados são armazenados com um nome de arquivo único
        run_simulation(turbines, env, simulation_params, r, ntheta, environment_params['Vinf'], num_turbines, turbine_params)

if __name__ == "__main__":
    runtest()

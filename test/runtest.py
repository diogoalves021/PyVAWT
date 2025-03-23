import sys
import os
import unittest
import numpy as np
import h5py
import matplotlib
matplotlib.use("TkAgg")  # Set a different backend
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import readaerodyn, Turbine, actuatorcylinder, Environment

atol = 1e-6

flnm = 'NACA_0012_mod'

#Parâmetros e variáveis de entrada

ntheta = 36

#Parâmetros da turbina
r = 35/2
twist = 0.0
delta = 0.0
af = readaerodyn(f'data/NACA_0012_mod.dat')
chord = 1.75 #solidity * r / B
B = 2
solidity = 0.1
centerX = 0
centerY = 0
Omega = 0.0

#Parâmetros de ambiente
Vinf = 2.0
rho = 1.225
mu = 1.7894e-5

#Digite 0 abaixo para avariar Vinf a cada iteração e manter Omega constante.
#Digite 1 abaixo para variar Omega a cada iteração e manter Vinf constante.
var_omega_vinf = 1 #0 ou 1 apenas

num_turbines = 2 #Only 1 or 2 turbines
 
turbines = [None] * 1 #Cria uma lista de tamanho 1 vazia
turbines[0] = Turbine(r, chord, twist, delta, B, af, Omega, 0.0, 0.0) #Inicializa a turbina

env = Environment(Vinf, rho, mu) #Cria o ambiente

class TestVAWTAC(unittest.TestCase):
    def test_one_turbine(self):
        self.assertIsInstance(turbines[0], Turbine)  # Teste: verifica se turbines[0] é uma instância de Turbine

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


    print('Simulação concluída.')

    #Salvando os resultados
    with h5py.File('results/single_unit_test_data.h5', 'w') as f:
        f.create_dataset("CPvec_old", data=CPvec)
        f.create_dataset("CTvec_old", data=CTvec)
        f.create_dataset("Rpvec_old", data=Rpvec)
        f.create_dataset("Tpvec_old", data=Tpvec)
        f.create_dataset("Zpvec_old", data=Zpvec)
        f.create_dataset("thetavec_old", data=thetavec)

    #Salvando em um arquivo .dat
    data_to_save = np.column_stack((tsrvec, CPvec, CTvec, Rpvec, Tpvec, Zpvec))

    header = "TSR CP CT Rp Tp Zp"
    np.savetxt("results/results.dat", data_to_save, header=header, fmt="%.6f", delimiter="\t")

    #Carregando os resultados
    with h5py.File('results/single_unit_test_data.h5', 'r') as f:
        CPvec_old = np.array(f["CPvec_old"])
        CTvec_old = np.array(f["CTvec_old"])
        Rpvec_old = np.array(f["Rpvec_old"])
        Tpvec_old = np.array(f["Tpvec_old"])
        Zpvec_old = np.array(f["Zpvec_old"])
        thetavec_old = np.array(f["thetavec_old"])

    #Teste de comparação
    assert np.allclose(CPvec, CPvec_old, atol=atol)
    assert np.allclose(CTvec, CTvec_old, atol=atol)
    assert np.allclose(Rpvec, Rpvec_old, atol=atol)
    assert np.allclose(Tpvec, Tpvec_old, atol=atol)
    assert np.allclose(Zpvec, Zpvec_old, atol=atol)
    assert np.allclose(thetavec, thetavec_old, atol=atol)

    print('\nGenerating results...')

    plt.figure(figsize=(10, 5))
    plt.plot(tsrvec, CPvec, color='blue', label='$C_p$')
    plt.title('Gráfico de $C_p$ x TSR ($\\lambda$)')
    plt.legend()
    plt.show()

#------------------------------
#Two Turbines
#------------------------------
elif num_turbines == 2:

    tsr = 3.5
    Omega = Vinf * tsr / r

    #Creating turbines
    turbines = [None] * 2
    turbines[0] = Turbine(r, chord, twist, delta, B, af, Omega, 0.0, 0.0)
    turbines[1] = Turbine(r, chord, twist, delta, B, af, -Omega, 0.0, 2 * r)

    # Verifique se são instâncias de Turbine e o atributo 'r'
    for turbine in turbines:
        print(f'Type of turbine: {type(turbine)}')
        print(f'Raio da turbina: {turbine.r}')

    #Calling actuator cylinder function
    CT, CP, Rp, Tp, Zp, theta = actuatorcylinder(turbines, env, ntheta)

    #Writing the HDF5 file
    with h5py.File('results/dual_init_test_data.h5', 'w') as file:
        file.create_dataset('CP_old', data=CP)
        file.create_dataset('CT_old', data=CT)
        file.create_dataset('Rp_old', data=Rp)
        file.create_dataset('Tp_old', data=Tp)
        file.create_dataset('Zp_old', data=Zp)
        file.create_dataset('theta_old', data=theta)

    #Reading the HDF5 file data
    with h5py.File('results/dual_init_test_data.h5', 'r') as file:
        CP_old = file['CP_old'][:]
        CT_old = file['CT_old'][:]
        Rp_old = file['Rp_old'][:]
        Tp_old = file['Tp_old'][:]
        Zp_old = file['Zp_old'][:]
        theta_old = file['theta_old'][:]


    #Testing (making sure the values are approximately equal)
    assert np.allclose(CP, CP_old, atol=atol)
    assert np.allclose(CT, CT_old, atol=atol)
    assert np.allclose(Rp, Rp_old, atol=atol)
    assert np.allclose(Tp, Tp_old, atol=atol)
    assert np.allclose(Zp, Zp_old, atol=atol)
    assert np.allclose(theta, theta_old, atol=atol)

    print('\nGenerating results...')

    plt.figure()
    plt.plot(theta, r * Tp[:, 0], label='Turbina 1') 
    plt.plot(theta, r * Tp[:, 1], label='Turbina 2')  
    plt.xlabel(r'$\theta$')
    plt.ylabel('Q (torque)')
    plt.xlim([0, 2 * np.pi])
    plt.legend()
    plt.show()

    # Salvar os dados em um arquivo .dat
    dados = np.column_stack((theta, r * Tp[:, 0], r * Tp[:, 1]))  # Combina as colunas em uma matriz
    np.savetxt('results/two_turbines_results.dat', dados, header="theta, torque_1, torque_2", comments='')

import sys
import os
import unittest
import numpy as np
import h5py
import matplotlib
matplotlib.use("TkAgg")  # Define um backend interativo
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
solidity = chord * B / r
centerX = 0
centerY = 0
Omega = 0.0

#Parâmetros de ambiente
Vinf = 0.0
rho = 1.225
mu = 1.7894e-5

#Digite 0 abaixo para avariar Vinf a cada iteração e manter Omega constante.
#Digite 1 abaixo para variar Omega a cada iteração e manter Vinf constante.
var_omega_vinf = 1 #0 ou 1 apenas
 
turbines = [None] * 1 #Cria uma lista de tamanho 1 vazia
turbines[0] = Turbine(r, chord, twist, delta, B, af, Omega, 0.0, 0.0) #Inicializa a turbina

env = Environment(Vinf, rho, mu) #Cria o ambiente

class TestVAWTAC(unittest.TestCase):
    def test_one_turbine(self):
        self.assertIsInstance(turbines[0], Turbine)  # Teste: verifica se turbines[0] é uma instância de Turbine

#------------------------
#Simulação de uma turbina
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

#----------------------------------------------------
#Visualização dos resultados
#----------------------------------------------------
print('\nGerando resultados...')

config_graficos = 0 #0 para gŕafico de cp; 1 para vários gráficos

if config_graficos == 0:
    plt.figure(figsize=(10, 5))
    plt.plot(tsrvec, CPvec, color='blue', label='$C_p$')
    plt.title('Gráfico de $C_p$ x TSR ($\\lambda$)')
    plt.legend()
    plt.show()


else:
    plt.figure(figsize=(10, 5))
    plt.suptitle(f'Gráficos do Perfil {flnm}')
    #plt.suptitle(f'Gráfico do Perfil {}')

    plt.subplot(2, 2, 1)
    plt.plot(tsrvec, CPvec, color='blue', label='$C_p$')
    plt.title('Gráfico de $C_p$ x TSR ($\\lambda$)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(tsrvec, CTvec, color='red', label='$C_t$')
    plt.title('Gráfico de $C_t$ x TSR ($\lambda$)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(tsrvec, Rpvec, color='green', label='$R_p$')
    plt.title('Gráfico de $R_p$ x TSR ($\lambda$)')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(tsrvec, Tpvec, color='orange', label='$T_p$')
    plt.title('Gráfico de $T_p$ x TSR ($\lambda$)')
    plt.legend()

    plt.tight_layout()
    print('\nPronto.\n')
    print('--' * 12)
    plt.show()
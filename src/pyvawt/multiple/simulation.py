import math
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root
from typing import Callable, Tuple

#Coeficientes de influência

def panelIntegration(xvec, yvec, thetavec, ifunc):
    """
    Executar integração dos paineis para influencia de coeficientes.
    Aplica para ambos Ay e Dx dependendo da função passada.
    """
    #Inicializar
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] #Assume angulos igualmente espaçados
    A = np.zeros((nx, ntheta))

    for i in range(nx):
        #Redefine a função para uso na integração
        def integrand(phi):
            return ifunc(xvec[i], yvec[i], phi)
        
        for j in range(ntheta):
            #Executar integração adaptativa
            result, error = quad(
                integrand,
                thetavec[j] -dtheta / 2,
                thetavec[j] + dtheta / 2,
                epsabs=1e-10
            )
            A[i, j] = result
            
    return A

def Dxintegrand(x, y, phi):
    """
    Integrando usado para computar Dx
    """
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)

    #print(v1, v2)
    #v1 e v2 não podem ser zero porque nunca integramos self. RxII lida com essa situação.
    return (v1 * math.sin(phi) - v2 * math.cos(phi)) / (2 * math.pi * (v1 * v1 + v2 * v2))

def Ayintegrand(x, y, phi):
    """
    Integrando usado para computar Ay
    """
    v1 = x + math.sin(phi)
    v2 = y - math.cos(phi)
    if abs(v1) < 1e-12 and abs(v2) < 1e-12:
        #Ocorre quando integramos self; função simétrica função simétrica em torno da singularidade deve integrar-se a zero
        return 0.0
    return (v1 * math.cos(phi) + v2 * math.sin(phi)) / (2 * math.pi * (v1 * v1 + v2 * v2))

def AyIJ(xvec, yvec, thetavec):
    """
    Computar AyIJ integrando com a função AyIntegrand
    """
    return panelIntegration(xvec, yvec, thetavec, Ayintegrand)

def DxIJ(xvec, yvec, thetavec):
    """
    Computar DxIJ integrando com a função Dxintegrand
    """
    return panelIntegration(xvec, yvec, thetavec, Dxintegrand)

def WxIJ(xvec, yvec, thetavec):
    #Inicializa
    nx = len(xvec)
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] #Supõe valores igualmente espaçados
    Wx = np.zeros((nx, ntheta))

    for i in range(nx):
        if (
            -1.0 <= yvec[i] <= 1.0
            and xvec[i] >= 0.0
            and xvec[i]**2 + yvec[i]**2 >= 1.0
        ):
            thetak = np.arccos(yvec[i])
            k = np.searchsorted(thetavec + dtheta / 2, thetak, side='right') #Índice da interseção
            if 0 <= k < ntheta:
                Wx[i, k] = -1.0
                Wx[i, ntheta - k - 1] = 1.0

    return Wx

def DxII(thetavec):
    #Inicializa
    ntheta = len(thetavec)
    dtheta = thetavec[1] - thetavec[0] #Supõe valores igualmente espaçados
    Rx = (dtheta / (4 * np.pi)) * np.ones((ntheta, ntheta))

    for i in range(ntheta):
        if i < ntheta // 2:
            Rx[i, i] = (-1 + 1.0 / ntheta) / 2.0
        else:
            Rx[i, i] = (1 + 1.0 / ntheta) / 2.0

    return Rx

def WxII(thetavec):
    #Inicializar
    ntheta = len(thetavec)
    Wx = np.zeros((ntheta, ntheta))

    for i in range(ntheta // 2, ntheta):
        Wx[i, ntheta - 1 - i] = -1

    return Wx

def precomputeMatrices(ntheta, modulepath):
    #Configurar discretização
    dtheta = 2 * np.pi / ntheta
    theta = np.arange(dtheta / 2, 2 * np.pi, dtheta)

    #Pré-computar matrizes de auto-influência
    Dxself = DxII(theta)
    Wxself = WxII(theta)
    Ayself = AyIJ(-np.sin(theta), np.cos(theta), theta)

    #Escrever no arquivo HDF5
    filepath = f'{modulepath}/theta-{ntheta}.h5'
    with h5py.File(filepath, 'w') as file:
        file.create_dataset('theta', data=theta)
        file.create_dataset('Dx', data=Dxself)
        file.create_dataset('Wx', data=Wxself)
        file.create_dataset('Ay', data=Ayself)
     
    return filepath


def matrixAssemble(centerX, centerY, radii, ntheta):
    """
    Monta as matrizes globais para as turbinas VAWT.

    Parâmetros:
        centerX, centerY: arrays com as coordenadas x e y dos centros das turbinas.
        radii: Array om os raios das turbinas.
        ntheta: número de divisões angulares.
    
    Retorna:
        Ax, Ay, theta: matrizes globais Ax e Ay e o vetor theta.
    """

    #Verificar e carregar o arquivo precomputado

    file = f'theta-{ntheta}.h5'
    modulepath = os.getcwd() #utiliza o diretório atual como caminho
    if not os.path.isfile(file):
        filepath = precomputeMatrices(ntheta, modulepath)
    else:
        filepath = os.path.join(modulepath, file)

    #ler dados do arquivo HDF5

    with h5py.File(filepath, 'r') as f:
        theta = f['theta'][:]
        Dxself = f['Dx'][:]
        Wxself = f['Wx'][:]
        Ayself = f['Ay'][:]

    #Inicializar as matrizes globais

    nturbines = len(radii)
    Dx = np.zeros((nturbines * ntheta, nturbines * ntheta))
    Wx = np.zeros((nturbines * ntheta, nturbines * ntheta))
    Ay = np.zeros((nturbines * ntheta, nturbines * ntheta))

    #Iterar sobre todas as turbinas
    for I in range(nturbines):
        for J in range(nturbines):
            #Coordenadas normalizadas em relação ao centro da turbina J
            x = (centerX[I] - radii[I] * np.sin(theta) - centerX[J]) / radii[J]
            y = (centerY[I] + radii[I] * np.cos(theta) - centerY[J]) / radii[J]

            #Auto-influência pré-computada
            if I == J:
                Dxsub = Dxself
                Wxsub = Wxself
                Aysub = Ayself

            #Pares com o mesmo raio ja mapeados
            elif J < I and radii[I] == radii[J]:
                Dxsub = Dx[J * ntheta:(J + 1) * ntheta, I * ntheta:(I + 1) * ntheta]
                Aysub = Ay[J * ntheta:(J + 1) * ntheta, I * ntheta:(I + 1) * ntheta]

                #Recalcular o termo de esteira
                Wxsub = WxIJ(x, y, theta)

            else:
                Dxsub = DxIJ(x, y, theta)
                Wxsub = WxIJ(x, y, theta)
                Aysub = AyIJ(x, y, theta)

            #Montar as submatrizes nas matrizes globais
            Dx[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Dxsub
            Wx[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Wxsub
            Ay[I * ntheta:(I + 1) * ntheta, J * ntheta:(J + 1) * ntheta] = Aysub

    #calcular Ax
    Ax = Dx + Wx

    return Ax, Ay, theta

#---------------------------------------
#
#-------- Coeficientes de força --------

class Turbine:
    def __init__(self, r: float, chord: float, twist: float, delta: float, B: int,
                af: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], Omega: float,
                centerX: float, centerY: float):

        self.r = r
        self.chord = chord
        self.twist = twist
        self.delta = delta
        self.B = B
        self.af = af  #Função que retorna cl e cd
        self.Omega = Omega
        self.centerX = centerX
        self.centerY = centerY

class Environment:
    def __init__(self, Vinf: float, rho: float, mu: float):
        self.Vinf = Vinf
        self.rho = rho
        self.mu = mu

def fast_trapz(y, x):
    """
    Integração trapezoidal 1D ultra-rápida usando produto escalar.
    Evita o overhead de validação de eixos/dimensões do numpy/scipy.
    """
    return 0.5 * np.dot(y[:-1] + y[1:], np.diff(x))

def radialforce(uvec, vvec, thetavec, turbine: Turbine, env: Environment):
    """
    Calcula forças radiais e coeficientes aerodinâmicos.
    """
    # Desempacotando os parâmetros da turbina e do ambiente
    r = turbine.r
    chord = turbine.chord
    twist = turbine.twist
    delta = turbine.delta
    B = turbine.B
    Omega = turbine.Omega
    Vinf = env.Vinf
    rho = env.rho

    # Direção de rotação
    rotation = np.sign(Omega)

    # Componentes de velocidade e ângulos
    Vn = Vinf * (1.0 + uvec) * np.sin(thetavec) - Vinf * vvec * np.cos(thetavec)
    Vt = (rotation * (Vinf * (1.0 + uvec) * np.cos(thetavec) + Vinf * vvec * np.sin(thetavec)) + abs(Omega) * r)
    W = np.sqrt(Vn**2 + Vt**2)
    phi = np.arctan2(Vn, Vt)
    alpha = phi - turbine.twist

    # Coeficientes aerodinâmicos (cl, cd) a partir do perfil
    cl, cd = turbine.af(alpha)

    # Rotação dos coeficientes de força
    cn = cl * np.cos(phi) + cd * np.sin(phi)
    ct = cl * np.sin(phi) - cd * np.cos(phi)

    # Força radial
    sigma = B * chord / r
    q = sigma / (4 * np.pi) * cn * (W / Vinf)**2

    # Forças instantâneas
    qdyn = 0.5 * rho * W**2
    Rp = -cn * qdyn * chord
    Tp = ct * qdyn * chord / np.cos(delta)
    Zp = -cn * qdyn * chord * np.tan(delta)

    # Fator de correção não linear
    integrand = (W / Vinf)**2 * (cn * np.sin(thetavec) - rotation * ct * np.cos(thetavec) / np.cos(delta))
    
    # MODIFICADO: Substituído np.trapz por fast_trapz
    CT = sigma / (4 * np.pi) * fast_trapz(integrand, thetavec)

    if CT > 2.0:
        a = 0.5 * (1.0 + np.sqrt(1.0 + CT))
        ka = 1.0 / (a - 1)
    elif CT > 0.96:
        a = 1.0 / 7 * (1 + 3.0 * np.sqrt(7.0 / 2 * CT - 3))
        ka = 18.0 * a / (7 * a**2 - 2 * a + 4)
    else:
        a = 0.5 * (1 - np.sqrt(1.0 - CT))
        ka = 1.0 / (1 - a)
    
    # Coeficiente de potência
    H = 1.0 # Altura por unidade
    Sref = 2 * r * H
    Q = r * Tp
    
    # MODIFICADO: Substituído np.trapz por fast_trapz
    P = abs(Omega) * B / (2 * np.pi) * fast_trapz(Q, thetavec)
    CP = P / (0.5 * rho * Vinf**3 * Sref)

    return q, ka, CT, CP, Rp, Tp, Zp

#------------------------------------
#
#-------- Resolver o Sistema --------

def residual(w, A, theta, k, turbines, env):
    #Configutação inicial
    ntheta = len(theta)
    nturbines = int(len(w) / (2 * ntheta))
    q = np.zeros(ntheta * nturbines)
    ka = 0.0

    for i in range(1, nturbines + 1):
        idx = slice((i - 1) * ntheta, i * ntheta)
        u = w[idx]

        idx_v = slice(ntheta * nturbines + (i - 1) * ntheta, ntheta * nturbines + i * ntheta)
        v = w[idx_v]

        q[idx], ka, *_ = radialforce(u, v, theta, turbines[i - 1], env)

    if nturbines == 1: #Se houber apenas uma turbina, usa o k da análise
        k = np.array([ka])
    
    kmult = np.repeat(k, ntheta)
    kmult = np.concatenate([kmult, kmult])

    return (A @ q) * kmult - w

def actuatorcylinder(turbines, env, ntheta, w0=None):
    # List comprehensions para extração geométrica
    centerX = np.array([turbine.centerX for turbine in turbines])
    centerY = np.array([turbine.centerY for turbine in turbines])
    radii = np.array([turbine.r for turbine in turbines])

    # Montar matrizes de indução globais
    Ax, Ay, theta = matrixAssemble(centerX, centerY, radii, ntheta)

    # Configuração inicial de dimensões
    ntheta = len(theta)
    nturbines = len(turbines)
    tol = 1e-6
    
    # Prealocação de arrays de saída
    CT = np.zeros(nturbines)
    CP = np.zeros(nturbines)
    Rp = np.zeros((ntheta, nturbines))
    Tp = np.zeros((ntheta, nturbines))
    Zp = np.zeros((ntheta, nturbines))
    k = np.zeros(nturbines)

    # =========================================================================
    # 🔴 OTIMIZAÇÃO: ESTRUTURA CONDICIONAL DO WARM START
    # =========================================================================
    
    # Se já temos o Warm Start do TSR anterior E o sistema possui múltiplas turbinas:
    # PULAMOS o solver individual de cada turbina inteiramente!
    if w0 is not None and nturbines > 1:
        w0_coupled = w0
        # Apenas calculamos explicitamente o fator k inicial (avaliação direta, sem root solver)
        for i in range(nturbines):
            u_start = w0[i * ntheta : (i + 1) * ntheta]
            v_start = w0[ntheta * nturbines + i * ntheta : ntheta * nturbines + (i + 1) * ntheta]
            _, k[i], *_ = radialforce(u_start, v_start, theta, turbines[i], env)
            
    else:
        # CASO CONTRÁRIO: Primeira iteração (w0=None) OU caso de uma única turbina (nturbines=1)
        w0_coupled = np.zeros(nturbines * ntheta * 2) if w0 is None else w0
        
        for i in range(nturbines):
            # Se for caso mono-turbina com warm start, o w0_single é o próprio w0 recebido
            if w0 is not None and nturbines == 1:
                w0_single = w0
            else:
                w0_single = np.zeros(ntheta * 2)

            idx = np.arange(i * ntheta, (i + 1) * ntheta)

            # Definir o resíduo para o problema de uma única turbina isolada
            def resid_single(x):
                return residual(
                    x,
                    np.vstack([Ax[idx][:, idx], Ay[idx][:, idx]]),
                    theta,
                    [1.0],
                    [turbines[i]],
                    env
                )
            
            # Resolve o sistema individual
            result = root(resid_single, w0_single, method='lm', tol=tol)
            w_single = result.x
            if not result.success:
                print(f'Solver não convergiu para a turbina {i + 1}. Mensagem: {result.message}')

            # Separar componentes obtidas e extrair o fator k
            u = w_single[:ntheta]
            v = w_single[ntheta:]
            _, k[i], CT[i], CP[i], Rp[:, i], Tp[:, i], Zp[:, i] = radialforce(u, v, theta, turbines[i], env)
            
            # Se for a primeira iteração de um caso multi-turbina, montamos o w0_coupled inicial
            if nturbines > 1:
                w0_coupled[i * ntheta : (i + 1) * ntheta] = u
                w0_coupled[ntheta * nturbines + i * ntheta : ntheta * nturbines + (i + 1) * ntheta] = v

        # Se o parque contiver apenas uma turbina, o problema físico termina aqui
        if nturbines == 1:
            return CT, CP, Rp, Tp, Zp, theta, w_single

    # ==========================================
    # Resolver sistema acoplado (Apenas para nturbines > 1)
    # ==========================================
    def resid_multiple(x):
        return residual(x, np.vstack([Ax, Ay]), theta, k, turbines, env)
    
    # Executa o solver acoplado global aproveitando o Warm Start direto (w0_coupled)
    result = root(resid_multiple, w0_coupled, method='lm', tol=tol)
    w_coupled = result.x
    if not result.success:
        print(f'Solver não convergiu para o sistema acoplado. Mensagem: {result.message}')
    
    # Pós-processamento final para extração dos coeficientes acoplados
    for i in range(nturbines): 
        idx = list(range(i * ntheta, (i + 1) * ntheta)) 

        u = w_coupled[idx]
        v = w_coupled[ntheta * nturbines + np.array(idx)]

        _, _, CT[i], CP[i], Rp[:, i], Tp[:, i], Zp[:, i] = radialforce(u, v, theta, turbines[i], env)

    return CT, CP, Rp, Tp, Zp, theta, w_coupled

#------------------------------------
#-------- Métodos Auxiliares --------

#Integração trapezoidal
def trapz(x, y):
    integral = 0.0
    for i in range(len(x) - 1):
        integral += (x[i+1] - x[i]) * 0.5 * (y[i] + y[i+1])
    return integral

#Integração para uma função periódica onde os pontos finais não alcançam os fins
def pInt(theta, f):
    #Computar integração trapezoidal
    integral = trapz(theta, f)

    #Adicione a contribuição dos pontos finais periódicos
    dtheta = 2 * theta[0] #Assume espaçamento igual, começa em 0
    integral += dtheta * 0.5 * (f[0] + f[-1])

    return integral

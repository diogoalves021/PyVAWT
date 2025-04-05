import numpy as np
from scipy.interpolate import UnivariateSpline

def readaerodyn(filename):
    """Apenas le um numero de Reynolds caso haja mais de um."""
    alpha = []
    cl = []
    cd = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Detecta automaticamente o tipo de arquivo:
    # Se nas primeiras 15 linhas houver a palavra "alpha" (em qualquer caixa) e não houver "EOT", é provável que seja o formato QBlade.
    if any('alpha' in line.lower() for line in lines[:15]) and not any('EOT' in line for line in lines):
        is_qblade = True
    else:
        is_qblade = False

    # Seleciona as linhas de dados conforme o formato detectado
    if is_qblade:
        # Formato QBlade: pula as 12 primeiras linhas
        data_lines = lines[12:]
    else:
        # Formato .dat: pula as 13 primeiras linhas e lê até encontrar "EOT"
        data_lines = []
        for line in lines[13:]:
            if 'EOT' in line:
                break
            data_lines.append(line)

    # Processa as linhas de dados
    for line in data_lines:
        parts = line.split()
        if len(parts) < 3:
            continue  # Ignora linhas com menos de 3 valores
        try:
            alpha.append(float(parts[0]))
            cl.append(float(parts[1]))
            cd.append(float(parts[2]))
        except ValueError:
            continue  # Ignora linhas que não podem ser convertidas para float
    
    # Converte listas para arrays do numpy e converte alpha de graus para radianos
    alpha = np.array(alpha) * np.pi / 180
    cl = np.array(cl)
    cd = np.array(cd)
    
    # Cria interpoladores spline para Cl e Cd
    afcl = UnivariateSpline(alpha, cl, s=0.1)
    afcd = UnivariateSpline(alpha, cd, s=0.001)

    '''
    with open(filename, 'r') as f:
        #pula as primeiras 13 linhas
        for _ in range(13):
            next(f)
        
        #Le os dados até encontrar "EOT"
        for line in f:
            if 'EOT' in line:
                break
            parts = line.split()
            alpha.append(float(parts[0]))
            cl.append(float(parts[1]))
            cd.append(float(parts[2]))

    #Converte as listas para arrays do numpy
    alpha = np.array(alpha) * np.pi / 180 #Converte graus para radianos
    cl = np.array(cl)
    cd = np.array(cd)

    #Cria interpolações spline de 1D (ignorando dependencia de Re)
    afcl = UnivariateSpline(alpha, cl, s=0.1)
    afcd = UnivariateSpline(alpha, cd, s=0.001)
    '''
    def af(alpha):
        """Retorna cl e cd interpolados para um dado alpha"""
        return afcl(alpha), afcd(alpha)
        
    return af

#af = readaerodyn('data/NACA_0012_mod.dat')
print('--' * 12)
print('\nInicializando a simulação...\n')
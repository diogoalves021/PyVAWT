import numpy as np
import matplotlib.pyplot as plt
from src.pyvawt.single.data_generation import get_cl_cd_neuralfoil, load_config
#from src.pyvawt.simulation import Turbine, Environment
import numba as nb

# ==============================================================================
# FUNÇÕES AUXILIARES ESCALARES (COMPILADAS EM C VIA NUMBA)
# ==============================================================================

@nb.njit(fastmath=True)
def abs_smooth(x, eps=1e-4):
    '''
    Aproximação suave do valor absoluto para evitar descontinuidades de derivada.
    '''
    return np.sqrt(x * x + eps * eps)


@nb.njit(fastmath=True)
def ksmin2(a, b, k=300.0):
    '''
    Mínimo suave (KS) para 2 escalares sem alocação de memória e estável numericamente.
    '''
    m = a if a < b else b
    return m - np.log(np.exp(-k * (a - m)) + np.exp(-k * (b - m))) / k


@nb.njit(fastmath=True)
def ksmax2(a, b, k=300.0):
    '''
    Máximo suave (KS) para 2 escalares sem alocação de memória e estável numericamente.
    '''
    m = a if a > b else b
    return m + np.log(np.exp(k * (a - m)) + np.exp(k * (b - m))) / k


@nb.njit(fastmath=True)
def interp2d_scalar(alpha_val, W_val, alpha_grid, W_grid, table):
    '''
    Interpolação bilinear escalar ultrarrápida para obter Cl/Cd da tabela estática.
    '''
    n_alpha = len(alpha_grid)
    n_w = len(W_grid)
    
    d_alpha = alpha_grid[1] - alpha_grid[0]
    d_w = W_grid[1] - W_grid[0]
    
    alpha_min, alpha_max = alpha_grid[0], alpha_grid[-1]
    w_min, w_max = W_grid[0], W_grid[-1]

    # Mapeia alpha para [-pi, pi]
    a = (alpha_val + np.pi) % (2.0 * np.pi) - np.pi
    w = W_val

    if a <= alpha_min:
        ia = 0; u = 0.0
    elif a >= alpha_max:
        ia = n_alpha - 2; u = 1.0
    else:
        pos_a = (a - alpha_min) / d_alpha
        ia = int(pos_a)
        u = pos_a - ia

    if w <= w_min:
        iw = 0; v = 0.0
    elif w >= w_max:
        iw = n_w - 2; v = 1.0
    else:
        pos_w = (w - w_min) / d_w
        iw = int(pos_w)
        v = pos_w - iw

    f00 = table[ia, iw]
    f10 = table[ia + 1, iw]
    f01 = table[ia, iw + 1]
    f11 = table[ia + 1, iw + 1]

    return (1.0 - u) * (1.0 - v) * f00 + u * (1.0 - v) * f10 + (1.0 - u) * v * f01 + u * v * f11


# ==============================================================================
# KERNEL PRINCIPAL DO BOEING-VERTOL (100% NUMBA NOPYTHON)
# ==============================================================================

@nb.njit(fastmath=True)
def boeing_vertol_jit(
    CL, CD, CM,
    alpha, adotnorm, umach, W,
    aoaStallPos, aoaStallNeg, AOA0, tc,
    BV_DynamicFlagL, BV_DynamicFlagD,
    alpha_grid, W_grid, cl_table, cd_table
):
    '''
    Núcleo dinâmico de Boeing-Vertol compilado sem nenhuma dependência de objetos Python.
    '''
    k1pos = 0.5
    k1neg = 0.5
    diff = 0.06 - tc
    smachl = 0.4 + 5.0 * diff
    hmachl = 0.9 + 2.5 * diff
    gammaxl = 1.4 - 6.0 * diff
    dgammal = gammaxl / (hmachl - smachl)
    smachm = 0.2
    hmachm = 0.7 + 2.5 * diff
    gammaxm = 1.0 - 2.5 * diff
    dgammam = gammaxm / (hmachm - smachm)

    # Limites de referência para alpha
    Fac = 0.9
    val_pos = abs_smooth(aoaStallPos - AOA0)
    val_neg = abs_smooth(aoaStallNeg - AOA0)
    
    dalphaRefMax = Fac * ksmin2(val_pos, val_neg) / ksmax2(k1pos, k1neg)
    TransA = 0.5 * dalphaRefMax
    sign_adot = 1.0 if adotnorm >= 0.0 else -1.0

    # --- Modelo de Sustentação (Lift) ---
    gammal = gammaxl - (umach - smachl) * dgammal
    dalphaLRef = gammal * np.sqrt(abs_smooth(adotnorm))
    dalphaLRef = ksmin2(dalphaLRef, dalphaRefMax)

    if adotnorm * (alpha - AOA0) < 0.0:
        dalphaL = k1neg * dalphaLRef
        alrefL = alpha - dalphaL * sign_adot
        if BV_DynamicFlagL == 1 and (aoaStallNeg < alrefL < aoaStallPos):
            BV_DynamicFlagL = 0
    else:
        dalphaL = k1pos * dalphaLRef
        alrefL = alpha - dalphaL * sign_adot
        if alpha <= aoaStallNeg or alpha >= aoaStallPos:
            BV_DynamicFlagL = 1
        else:
            BV_DynamicFlagL = 0

    # --- Modelo de Arraste (Drag) ---
    gammam = gammaxm - (umach - smachm) * dgammam
    if umach < smachm:
        gammam = gammaxm

    dalphaDRef = gammam * np.sqrt(abs_smooth(adotnorm))
    dalphaDRef = ksmin2(dalphaDRef, dalphaRefMax)

    if adotnorm * (alpha - AOA0) < 0.0:
        dalphaD = k1neg * dalphaDRef
        alLagD = alpha - dalphaD * sign_adot
        if BV_DynamicFlagD == 1:
            delN = aoaStallNeg - alLagD
            delP = alLagD - aoaStallPos
        else:
            delN = 0.0
            delP = 0.0
    else:
        dalphaD = k1pos * dalphaDRef
        alLagD = alpha - dalphaD * sign_adot
        delN = aoaStallNeg - alpha
        delP = alpha - aoaStallPos

    if delN > TransA or delP > TransA:
        alrefD = alLagD
        BV_DynamicFlagD = 1
    elif 0.0 < delN < TransA:
        alrefD = alpha + (alLagD - alpha) * delN / TransA
        BV_DynamicFlagD = 1
    elif 0.0 < delP < TransA:
        alrefD = alpha + (alLagD - alpha) * delP / TransA
        BV_DynamicFlagD = 1
    else:
        alrefD = alpha
        BV_DynamicFlagD = 0

    # --- Correções de Stall Dinâmico ---
    if BV_DynamicFlagL == 1:
        CL_ref = interp2d_scalar(alrefL, W, alpha_grid, W_grid, cl_table)
        denom = (alrefL - AOA0)
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0 else -1e-6
        CL = CL_ref / denom * (alpha - AOA0)

    if BV_DynamicFlagD == 1:
        CD = interp2d_scalar(alrefD, W, alpha_grid, W_grid, cd_table)

    return CL, CD, CM, int(BV_DynamicFlagL), int(BV_DynamicFlagD)


# ==============================================================================
# WRAPPER PYTHON (COMPATIBILIDADE COM O RESTANTE DO CÓDIGO)
# ==============================================================================

def Boeing_Vertol(
    CL, CD, CM, alpha, adotnorm, umach, Re,
    aoaStallPos, aoaStallNeg, AOA0, tc,
    BV_DynamicFlagL, BV_DynamicFlagD,
    turbine, env, turbine_index, airfoil_index, family_factor=0.0
):
    '''
    Wrapper de compatibilidade. Extrai as tabelas pré-calculadas e dispara 
    o kernel compilado sem overhead.
    '''
    aero = turbine.aero
    W = Re * env.mu / (env.rho * turbine.chord)

    return boeing_vertol_jit(
        float(CL), float(CD), float(CM),
        float(alpha), float(adotnorm), float(umach), float(W),
        float(aoaStallPos), float(aoaStallNeg), float(AOA0), float(tc),
        int(BV_DynamicFlagL), int(BV_DynamicFlagD),
        aero.alpha_grid, aero.W_grid, aero.cl_table, aero.cd_table
    )

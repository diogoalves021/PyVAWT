import numpy as np
import scipy.io as sio

# Adaptar codigo para formato python: python nao le .mat igual ao MATLAB e no momento esta programado para ler como se fosse em matlab

def load_frame(frame):

    # Load experiment condition variables
    b = 0.61 / 2   # McAlister et al. (1982) - page 14
    a_inf = 340    # Assumed MSL conditions

    if 7019 <= frame <= 14220:
        airfoil = 'NACA0012'
    elif 24022 <= frame <= 31310:
        airfoil = 'AMES-01'
    elif frame >= 67000:
        airfoil = 'NLR-7301'
    else:
        airfoil = None

    # Load .mat file
    filepath = f"../NASA Data/frame_{frame}.mat"
    mat_data = sio.loadmat(filepath)

    # Extrair variáveis do .mat (se existirem)
    alpha_0         = mat_data.get('alpha_0', np.nan)
    delta_alpha     = mat_data.get('delta_alpha', np.nan)
    k               = mat_data.get('k', np.nan)
    alpha_exp_cl    = mat_data.get('alpha_exp_cl', np.nan)
    cl_exp          = mat_data.get('cl_exp', np.nan)
    alpha_exp_cm    = mat_data.get('alpha_exp_cm', np.nan)
    cm_exp          = mat_data.get('cm_exp', np.nan)
    alpha_exp_cd    = mat_data.get('alpha_exp_cd', np.nan)
    cd_exp          = mat_data.get('cd_exp', np.nan)

    # Outros experimentais default = nan
    alpha_exp_cn    = np.nan
    cn_exp          = np.nan
    alpha_exp_cc    = np.nan
    cc_exp          = np.nan

    # Se variáveis de tempo não existirem no .mat → nan
    time_exp_cl     = mat_data.get('time_exp_cl', np.nan)
    clt_exp         = mat_data.get('clt_exp', np.nan)
    time_exp_cm     = mat_data.get('time_exp_cm', np.nan)
    cmt_exp         = mat_data.get('cmt_exp', np.nan)
    time_exp_cd     = mat_data.get('time_exp_cd', np.nan)
    cdt_exp         = mat_data.get('cdt_exp', np.nan)

    # Autores (se existir no .mat)
    authors         = mat_data.get('authors', None)

    return (b, a_inf, airfoil,
            mat_data.get('M', np.nan),
            alpha_0, delta_alpha, k,
            alpha_exp_cl, cl_exp,
            alpha_exp_cm, cm_exp,
            alpha_exp_cd, cd_exp,
            alpha_exp_cn, cn_exp,
            alpha_exp_cc, cc_exp,
            time_exp_cl, clt_exp,
            time_exp_cm, cmt_exp,
            time_exp_cd, cdt_exp,
            authors)


def read_data(input_value, model):
    '''
    Reads reference/experimental data and sets up the aerodynamic parameters for BL/BLS models.

    Parameters
    ----------
    input_value : int, str, or list
        Input identifier (NASA frame, GUD experiment, other, or a test cell).
    model : str
        Either 'BL' or 'BLS'.

    Returns
    -------
    authors : list
        Authors of the experiment or reference (may contain NaN if test case).
    data : list
        Experimental and model data.
    params : dict
        Dictionary containing flow properties, airfoil parameters, and test conditions.    
    '''

    # Handle input
    if isinstance(input_value, (int, float)):
        if input_value > 1e6:
            GUD = input_value
        elif input_value >1000:
            frame = input_value
        else:
            other = input_value
    else:
        test = input_value

    # Load data from frame, run or test condition
    if 'frame' in locals():
        b, a_inf, airfoil, M, a_0, a_1, k, alpha_exp_cl, cl_exp, \
        alpha_exp_cm, cm_exp, alpha_exp_cd, cd_exp, alpha_exp_cn, cn_exp, \
        alpha_exp_cc, cc_exp, time_exp_cl, clt_exp, time_exp_cm, cmt_exp, \
        time_exp_cd, cdt_exp, authors = load_frame(frame)

        time_exp_cn = np.nan
        cnt_exp = np.nan
        time_exp_cc = np.nan
        cct_exp = np.nan
        alpha_mod_cl = cl_mod = alpha_mod_cm = cm_mod = alpha_mod_cd = cd_mod = \
        alpha_mod_cn = cn_mod = alpha_mod_cc = cc_mod = np.nan

    elif 'GUD' in locals():
        b, a_inf, airfoil, M, a_0, a_1, k, alpha_exp_cl, cl_exp, \
        alpha_exp_cm, cm_exp, alpha_exp_cd, cd_exp, alpha_exp_cn, cn_exp, \
        alpha_exp_cc, cc_exp, time_exp_cl, clt_exp, time_exp_cm, cmt_exp, \
        time_exp_cd, cdt_exp, authors, time_exp_cn, cnt_exp, time_exp_cc, cct_exp = load_GUD(GUD)

        alpha_mod_cl = cl_mod = alpha_mod_cm = cm_mod = alpha_mod_cd = cd_mod = \
        alpha_mod_cn = cn_mod = alpha_mod_cc = cc_mod = np.nan

    elif 'other' in locals():
        b, a_inf, airfoil, M, a_0, a_1, k, alpha_exp_cl, cl_exp, \
        alpha_exp_cn, cn_exp, alpha_exp_cm, cm_exp, alpha_exp_cd, cd_exp, \
        alpha_exp_cc, cc_exp, alpha_mod_cl, cl_mod, alpha_mod_cn, cn_mod, \
        alpha_mod_cm, cm_mod, alpha_mod_cd, cd_mod, alpha_mod_cc, cc_mod, authors = load_other(other)
        
        time_exp_cl = clt_exp = time_exp_cm = cmt_exp = time_exp_cd = cdt_exp = \
        time_exp_cn = cnt_exp = time_exp_cc = cct_exp = np.nan

    elif 'test' in locals():
        # Standard values for b and a_inf if not provided
        if len(test) == 5:
            b = 0.61 / 2
            a_inf = 340
        elif len(test) == 6:
            b = test[5]
            a_inf = 340
        elif len(test) == 7:
            b = test[5]
            a_inf = test[6]
        else:
            raise ValueError('Set test as a list with 5 to 7 entries')
        
        # Set remaining test conditions
        M, k, a_0, a_1, airfoil = test[:5]
        authors = [np.nan]

        alpha_exp_cl = cl_exp = alpha_exp_cm = cm_exp = alpha_exp_cd = cd_exp = \
        alpha_exp_cn = cn_exp = alpha_exp_cc = cc_exp = np.nan

        alpha_mod_cl = cl_mod = alpha_mod_cm = cm_mod = alpha_mod_cd = cd_mod = \
        alpha_mod_cn = cn_mod = alpha_mod_cc = cc_mod = np.nan

        time_exp_cl = clt_exp = time_exp_cm = cmt_exp = time_exp_cd = cdt_exp = \
        time_exp_cn = cnt_exp = time_exp_cc = cct_exp = np.nan

    # --- Gather data
    data = [
        alpha_exp_cl, cl_exp, alpha_exp_cm, cm_exp, alpha_exp_cd, cd_exp,
        time_exp_cl, clt_exp, time_exp_cm, cmt_exp, time_exp_cd, cdt_exp,
        alpha_exp_cn, cn_exp, alpha_exp_cc, cc_exp,
        alpha_mod_cl, cl_mod, alpha_mod_cm, cm_mod, alpha_mod_cd, cd_mod,
        alpha_mod_cn, cn_mod, alpha_mod_cc, cc_mod,
        time_exp_cn, cnt_exp, time_exp_cc, cct_exp
    ]

    # --- Flow properties
    U = M * a_inf
    beta = np.sqrt(1 - M**2)

    # --- Airfoil parameters
    if model == "BL":
        params = airfoil_parameters(airfoil, M, U, b)
    elif model == "BLS":
        params = airfoil_parameters_BLS(airfoil, M, U, b)

    # --- Set all flow and test condition variables
    params['M'] = M
    params['U'] = U
    params['b'] = b
    params['a_inf'] = a_inf
    params['beta'] = beta
    params['a_0'] = a_0
    params['a_1'] = a_1
    params['k'] = k
    params['airfoil'] = airfoil

    return authors, data, params
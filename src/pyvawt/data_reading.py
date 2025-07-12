import numpy as np
from scipy.interpolate import UnivariateSpline

def readaerodyn(filename):
    """
    Reads an aerodynamic data file and returns a function that interpolates Cl and Cd 
    as a function of angle of attack (alpha).

    This function automatically detects the file format (QBlade or .dat) and extracts 
    angle of attack (in degrees), lift coefficient (Cl), and drag coefficient (Cd) data.
    Then, it creates spline interpolators for Cl and Cd.

    Parameters
    ----------
    filename : str
        Path to the file containing aerodynamic data.

    Returns
    -------
    af : function
        A function that takes an angle of attack in radians (float or array) and returns 
        the interpolated Cl and Cd values.
        Example: `cl, cd = af(alpha)`

    Notes
    -----
    - The angle of attack in the file is assumed to be in degrees and is internally 
      converted to radians.
    - Only one Reynolds number is considered, even if multiple are present.
    - Smoothed splines are used to reduce noise (s=0.1 for Cl, s=0.001 for Cd).
    """
    alpha = []
    cl = []
    cd = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Automatically detect file format:
    if any('alpha' in line.lower() for line in lines[:15]) and not any('EOT' in line for line in lines):
        is_qblade = True
    else:
        is_qblade = False

    # Select data lines based on format
    if is_qblade:
        data_lines = lines[11:]
    else:
        data_lines = []
        for line in lines[13:]:
            if 'EOT' in line:
                break
            data_lines.append(line)

    # Process data lines
    for line in data_lines:
        parts = line.split()
        if len(parts) < 3:
            continue 
        try:
            alpha.append(float(parts[0]))
            cl.append(float(parts[1]))
            cd.append(float(parts[2]))
        except ValueError:
            continue  
    
    # Convert lists to NumPy arrays and convert degrees to radians
    alpha = np.array(alpha) * np.pi / 180
    cl = np.array(cl)
    cd = np.array(cd)
    
    # Create 1D spline interpolators
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
        """
        Returns interpolated lift and drag coefficients for a given angle of attack.

        Parameters
        ----------
        alpha : float or array_like
            Angle of attack in radians.

        Returns
        -------
        cl : float or ndarray
            Interpolated lift coefficient.

        cd : float or ndarray
            Interpolated drag coefficient.
        """
        return afcl(alpha), afcd(alpha)
        
    return af

#af = readaerodyn('data/NACA_0012_mod.dat')
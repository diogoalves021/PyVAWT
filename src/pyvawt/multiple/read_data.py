import numpy as np
from scipy.interpolate import UnivariateSpline

def readaerodyn(filename):
    """Apenas le um numero de Reynolds caso haja mais de um."""
    alpha = []
    cl = []
    cd = []

    with open(filename, 'r') as f:
        #Ignore first 13 lines
        for _ in range(13):
            next(f)
        
        #Read data and stops when finding "EOT"
        for line in f:
            if 'EOT' in line:
                break
            parts = line.split()
            alpha.append(float(parts[0]))
            cl.append(float(parts[1]))
            cd.append(float(parts[2]))

    #Convert lists to numpy arrays
    alpha = np.array(alpha) * np.pi / 180 #Converting degrees to radians
    cl = np.array(cl)
    cd = np.array(cd)

    #Create 1D spline interpolation (ignoring Reynolds dependency)
    afcl = UnivariateSpline(alpha, cl, s=0.1)
    afcd = UnivariateSpline(alpha, cd, s=0.001)

    def af(alpha):
        """Retorna cl e cd interpolados para um dado alpha"""
        return afcl(alpha), afcd(alpha)
    
    return af


from pathlib import Path
from typing import Callable, Union
import numpy as np
from scipy.interpolate import UnivariateSpline

# Generic type to accept both single numbers (float) and vectors (np.ndarray)
ArrayOrFloat = Union[float, np.ndarray]


def readaerodyn(
    filename: Union[str, Path],
    s_cl: float = 0.1,
    s_cd: float = 0.001,
    skip_header: int = 13,
) -> Callable[[ArrayOrFloat], tuple[ArrayOrFloat, ArrayOrFloat]]:
    """
    Reads an AeroDyn airfoil polar file and returns an interpolating function
    (UnivariateSpline) for the lift (Cl) and drag (Cd) coefficients.

    Parameters
    ----------
    filename : str or Path
        Path to the AeroDyn airfoil polar file.
    s_cl : float, default=0.1
        Spline smoothing factor for Cl.
    s_cd : float, default=0.001
        Spline smoothing factor for Cd.
    skip_header : int, default=13
        Number of header lines to skip at the beginning of the file.

    Returns
    -------
    af : Callable[[ArrayOrFloat], tuple[ArrayOrFloat, ArrayOrFloat]]
        Callback function that takes the angle of attack alpha (in radians)
        and returns a tuple of interpolated (cl, cd) values.
    """
    filepath = Path(filename)

    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath.resolve()}")

    alpha_deg: list[float] = []
    cl_list: list[float] = []
    cd_list: list[float] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        # Skip header lines
        for _ in range(skip_header):
            next(f, None)

        # Read data until finding the "EOT" termination tag
        for line in f:
            if "EOT" in line:
                break

            parts = line.split()
            if len(parts) >= 3:
                alpha_deg.append(float(parts[0]))
                cl_list.append(float(parts[1]))
                cd_list.append(float(parts[2]))

    # Convert degrees to radians and convert to NumPy arrays
    alpha_rad = np.deg2rad(alpha_deg)
    cl_arr = np.array(cl_list, dtype=np.float64)
    cd_arr = np.array(cd_list, dtype=np.float64)

    # Ensure strictly increasing order (required by UnivariateSpline)
    sort_idx = np.argsort(alpha_rad)
    alpha_sorted = alpha_rad[sort_idx]
    cl_sorted = cl_arr[sort_idx]
    cd_sorted = cd_arr[sort_idx]

    # Build 1D splines maintaining exact original physics and smoothing factors
    afcl = UnivariateSpline(alpha_sorted, cl_sorted, s=s_cl)
    afcd = UnivariateSpline(alpha_sorted, cd_sorted, s=s_cd)

    def af(alpha: ArrayOrFloat) -> tuple[ArrayOrFloat, ArrayOrFloat]:
        """
        Returns interpolated cl and cd for a given angle of attack alpha (in radians).
        """
        return afcl(alpha), afcd(alpha)

    return af

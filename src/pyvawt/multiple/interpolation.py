"""
High-Performance Numerical Interpolation Module for PyVAWT.

Provides Numba JIT-compiled 2D bilinear interpolation kernels operating on
structured grids (angle of attack $\\alpha$ vs. logarithm of relative velocity $w$)
for fast aerodynamic coefficient evaluations.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _bilinear_interp_2d_numba(
    alpha_wrapped: np.ndarray,
    w_clamped: np.ndarray,
    values: np.ndarray,
    alpha_min: float,
    inv_dalpha: float,
    log_w_min: float,
    inv_dlog_w: float,
    n_alpha: int,
    n_w: int
) -> np.ndarray:
    """
    Compiled 2D bilinear interpolation kernel via Numba JIT.

    Performs fast, boundary-clamped bilinear interpolation over a 2D structured mesh.
    The grid coordinates assume uniform spacing in $\\alpha$ (angle of attack) and uniform
    spacing in $\\ln(w)$ (logarithmic relative velocity).

    Parameters
    ----------
    alpha_wrapped : np.ndarray
        Flattened or multidimensional array of target angle of attack ($\alpha$) values in radians.
    w_clamped : np.ndarray
        Array of target relative flow velocity ($w$) values in m/s, clamped within lookup table bounds.
    values : np.ndarray, shape (n_alpha, n_w)
        Pre-computed 2D table of physical scalar field values (e.g., $C_l$ or $C_d$).
    alpha_min : float
        Minimum boundary value of the $\\alpha$ grid coordinate ($\alpha_{\\min}$).
    inv_dalpha : float
        Inverse grid step size along the $\\alpha$ coordinate, defined as $(n_\\alpha - 1) / (\\alpha_{\\max} - \\alpha_{\\min})$.
    log_w_min : float
        Natural logarithm of the minimum relative velocity grid coordinate ($\ln w_{\\min}$).
    inv_dlog_w : float
        Inverse grid step size along the $\\ln(w)$ coordinate, defined as $(n_w - 1) / (\\ln w_{\\max} - \\ln w_{\\min})$.
    n_alpha : int
        Total number of grid points along the angle of attack axis.
    n_w : int
        Total number of grid points along the relative velocity axis.

    Returns
    -------
    out : np.ndarray
        Interpolated values reshaped to match the input shape of `alpha_wrapped`.
    """
    alpha_flat = alpha_wrapped.ravel()
    w_flat = w_clamped.ravel()
    n = alpha_flat.size
    out = np.empty(n, dtype=np.float64)

    # Upper boundary safety limits for array indexing
    max_i = n_alpha - 1.000001
    max_j = n_w - 1.000001

    for k in range(n):
        # Calculate continuous fractional index coordinates
        fi = (alpha_flat[k] - alpha_min) * inv_dalpha
        fj = (np.log(w_flat[k]) - log_w_min) * inv_dlog_w

        # Clamp fractional indices within bounds
        fi = max(0.0, min(fi, max_i))
        fj = max(0.0, min(fj, max_j))

        # Integer grid cell indices
        i0 = int(fi)
        j0 = int(fj)
        i1 = i0 + 1
        j1 = j0 + 1

        # Normalized cell interpolation weights
        t = fi - i0
        u = fj - j0

        # Corner node values from lookup matrix
        v00 = values[i0, j0]
        v10 = values[i1, j0]
        v01 = values[i0, j1]
        v11 = values[i1, j1]

        # Standard 2D bilinear interpolation formula
        out[k] = (1.0 - t) * (1.0 - u) * v00 + t * (1.0 - u) * v10 + (1.0 - t) * u * v01 + t * u * v11

    return out.reshape(alpha_wrapped.shape)


class FastBilinear2D:
    """
    Python interface wrapper for the Numba 2D bilinear interpolator.

    Pre-calculates and caches grid step sizes and logarithmic coordinate offsets
    to eliminate computational overhead during repetitive calls in numerical loops.

    Parameters
    ----------
    alpha_grid : np.ndarray
        1D linearly spaced vector of angle of attack grid coordinates in radians.
    w_grid : np.ndarray
        1D logarithmically spaced vector of relative velocity grid coordinates in m/s.
    values : np.ndarray, shape (n_alpha, n_w)
        2D matrix containing pre-computed grid values to interpolate.

    Attributes
    ----------
    n_alpha : int
        Number of grid points along the $\\alpha$ axis.
    n_w : int
        Number of grid points along the $w$ axis.
    values : np.ndarray
        C-contiguous 2D NumPy array of underlying data.
    alpha_min : float
        Lower boundary value of the $\\alpha$ axis.
    inv_dalpha : float
        Pre-calculated inverse step size along $\\alpha$.
    log_w_min : float
        Natural log of the lower boundary value of the $w$ axis.
    inv_dlog_w : float
        Pre-calculated inverse step size along $\\ln(w)$.
    """

    def __init__(self, alpha_grid: np.ndarray, w_grid: np.ndarray, values: np.ndarray) -> None:
        self.n_alpha = len(alpha_grid)
        self.n_w = len(w_grid)
        self.values = np.ascontiguousarray(values, dtype=np.float64)

        # Pre-compute alpha grid constants
        self.alpha_min = float(alpha_grid[0])
        self.inv_dalpha = float((self.n_alpha - 1) / (alpha_grid[-1] - alpha_grid[0]))

        # Pre-compute logarithmic velocity grid constants
        self.log_w_min = float(np.log(w_grid[0]))
        self.inv_dlog_w = float((self.n_w - 1) / (np.log(w_grid[-1]) - np.log(w_grid[0])))

    def __call__(self, alpha_wrapped: np.ndarray, w_clamped: np.ndarray) -> np.ndarray:
        """
        Evaluate the 2D bilinear interpolator at specified coordinate points.

        Parameters
        ----------
        alpha_wrapped : np.ndarray
            Target angle of attack ($\alpha$) values in radians, wrapped to $[-\pi, \pi]$.
        w_clamped : np.ndarray
            Target relative velocity ($w$) values in m/s, clamped to grid boundaries.

        Returns
        -------
        np.ndarray
            Interpolated numerical array with identical dimensions to `alpha_wrapped`.
        """
        return _bilinear_interp_2d_numba(
            alpha_wrapped,
            w_clamped,
            self.values,
            self.alpha_min,
            self.inv_dalpha,
            self.log_w_min,
            self.inv_dlog_w,
            self.n_alpha,
            self.n_w
        )

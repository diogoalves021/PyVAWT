"""
Numerical interpolation and JIT-compiled mathematical kernels for VAWT simulations.

Provides high-performance Numba-accelerated functions for fast 2D Look-Up Table (LUT)
bilinear interpolation and 1D numerical integration via the trapezoidal rule.
"""

from __future__ import annotations

import numba as nb
import numpy as np


# ==============================================================================
# NUMERICAL & INTERPOLATION KERNELS
# ==============================================================================

@nb.njit(fastmath=True, cache=True)
def interpolate_2d_lut(
    alpha_vec: np.ndarray,
    W_vec: np.ndarray,
    alpha_grid: np.ndarray,
    W_grid: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:
    """
    Perform fast 2D bilinear interpolation over a structured Lookup Table (LUT).

    Maps query angles of attack to the interval [-π, π] and evaluates table values
    using index bounding and uniform grid spacing.

    Parameters
    ----------
    alpha_vec : np.ndarray
        1D array of query angle of attack values [rad].
    W_vec : np.ndarray
        1D array of query relative velocity values [m/s].
    alpha_grid : np.ndarray
        1D array of uniformly spaced grid coordinates for angle of attack [rad].
    W_grid : np.ndarray
        1D array of uniformly spaced grid coordinates for relative velocity [m/s].
    table : np.ndarray
        2D matrix of shape `(len(alpha_grid), len(W_grid))` containing the
        precomputed values to interpolate (e.g., aerodynamic coefficients).

    Returns
    -------
    np.ndarray
        1D array of bilinearly interpolated values corresponding to each
        (`alpha_vec`, `W_vec`) pair.
    """
    n = len(alpha_vec)
    out = np.empty(n, dtype=np.float64)

    n_alpha = len(alpha_grid)
    n_w = len(W_grid)

    d_alpha = alpha_grid[1] - alpha_grid[0]
    d_w = W_grid[1] - W_grid[0]

    alpha_min, alpha_max = alpha_grid[0], alpha_grid[-1]
    w_min, w_max = W_grid[0], W_grid[-1]

    for i in range(n):
        # Map angle of attack to [-pi, pi]
        a = (alpha_vec[i] + np.pi) % (2.0 * np.pi) - np.pi
        w = W_vec[i]

        # Index search and bounding for alpha
        if a <= alpha_min:
            ia = 0
            u = 0.0
        elif a >= alpha_max:
            ia = n_alpha - 2
            u = 1.0
        else:
            pos_a = (a - alpha_min) / d_alpha
            ia = int(pos_a)
            u = pos_a - ia

        # Index search and bounding for relative velocity W
        if w <= w_min:
            iw = 0
            v = 0.0
        elif w >= w_max:
            iw = n_w - 2
            v = 1.0
        else:
            pos_w = (w - w_min) / d_w
            iw = int(pos_w)
            v = pos_w - iw

        # Bilinear interpolation
        f00 = table[ia, iw]
        f10 = table[ia + 1, iw]
        f01 = table[ia, iw + 1]
        f11 = table[ia + 1, iw + 1]

        out[i] = (
            (1.0 - u) * (1.0 - v) * f00
            + u * (1.0 - v) * f10
            + (1.0 - u) * v * f01
            + u * v * f11
        )

    return out


@nb.njit(fastmath=True, cache=True)
def fast_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute 1D numerical integration using the trapezoidal rule.

    Parameters
    ----------
    y : np.ndarray
        1D array of function values to integrate.
    x : np.ndarray
        1D array of sample points corresponding to `y`.

    Returns
    -------
    float
        Approximated integral value.
    """
    n = len(x)
    integral = 0.0
    for i in range(n - 1):
        integral += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return integral


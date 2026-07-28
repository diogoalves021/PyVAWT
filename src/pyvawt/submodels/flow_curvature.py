from typing import Any, Union
import numba as nb
import numpy as np


# ==============================================================================
# JIT-Compiled C Kernel
# ==============================================================================


@nb.njit(fastmath=True, cache=True)
def _corrected_flow_kernel(
    alpha: Union[float, np.ndarray],
    omega: float,
    relative_velocity: Union[float, np.ndarray],
    chord: float,
    normalized_hook_point: float,
) -> Union[float, np.ndarray]:
    """
    Fast C-compiled kernel for flow curvature angle of attack correction.

    Calculates: alpha_corr = alpha - (omega * chord / W) * (normalized_hook_point + 0.25)
    Handles both scalar values and 1D arrays with safe zero-division handling.
    """
    factor = omega * chord * (normalized_hook_point + 0.25)

    # Vetores (Arrays 1D)
    if np.ndim(alpha) > 0:
        n = len(alpha)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            w_val = relative_velocity[i]
            w_safe = w_val if abs(w_val) > 1e-6 else 1e-6
            out[i] = alpha[i] - (factor / w_safe)
        return out

    # Escalares
    else:
        w_safe = relative_velocity if abs(relative_velocity) > 1e-6 else 1e-6
        return alpha - (factor / w_safe)


# ==============================================================================
# Model Classes (Interface Externa Preservada)
# ==============================================================================


class FlowCurvatureModel:
    """
    Flow Curvature Model backed by Numba JIT compilation.
    """

    def __init__(self, chord: float, normalized_hook_point: float = 0.0):
        self.chord = float(chord)
        self.normalized_hook_point = float(normalized_hook_point)

    def corrected_flow(
        self,
        alpha: Union[float, np.ndarray],
        omega: float,
        relative_velocity: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Retorna a curvatura do fluxo para um dado alpha.
        """
        # Garante tipos primitivos/arrays C-contiguous para o Numba
        alpha_arr = (
            np.asarray(alpha, dtype=np.float64)
            if isinstance(alpha, (list, np.ndarray))
            else float(alpha)
        )
        w_arr = (
            np.asarray(relative_velocity, dtype=np.float64)
            if isinstance(relative_velocity, (list, np.ndarray))
            else float(relative_velocity)
        )

        return _corrected_flow_kernel(
            alpha_arr,
            float(omega),
            w_arr,
            self.chord,
            self.normalized_hook_point,
        )


class FlowCurvatureManager:
    """
    Simple manager that wraps FlowCurvatureModel and provides an enabled flag.
    """

    def __init__(
        self,
        chord: float,
        normalized_hook_point: float = 0.0,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled)
        self.model = FlowCurvatureModel(
            chord=chord, normalized_hook_point=normalized_hook_point
        )

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def corrected_flow(
        self,
        alpha: Union[float, np.ndarray],
        omega: float,
        relative_velocity: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Apply flow curvature correction if enabled.

        alpha: scalar or numpy array
        omega: scalar
        relative_velocity: scalar or numpy array (broadcastable to alpha)
        """
        if not self.enabled:
            return alpha

        return self.model.corrected_flow(alpha, omega, relative_velocity)

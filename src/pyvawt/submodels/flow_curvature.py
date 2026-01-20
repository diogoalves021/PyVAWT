class FlowCurvatureModel:
    def __init__(self, chord, normalized_hook_point = 0.0):
        self.chord = chord
        self.normalized_hook_point = normalized_hook_point

    def corrected_flow(self, alpha, omega, relative_velocity):
        """Retorna a curvatura do fluxo para um dado alpha"""
        return alpha - omega * self.chord / relative_velocity * (
            self.normalized_hook_point + 0.25
        )

class FlowCurvatureManager:
    """
    Simple manager that wraps FlowCurvatureModel and provides an enabled flag.
    """
    def __init__(self, chord, normalized_hook_point: float = 0.0, enabled: bool = True):
        self.enabled = enabled
        self.model = FlowCurvatureModel(chord=chord, normalized_hook_point=normalized_hook_point)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def corrected_flow(self, alpha, omega, relative_velocity):
        """
        Apply flow curvature correction if enabled.
        alpha: scalar or numpy array
        omega: scalar
        relative_velocity: scalar or numpy array (broadcastable to alpha)
        """
        if not self.enabled:
            return alpha
        return self.model.corrected_flow(alpha, omega, relative_velocity)
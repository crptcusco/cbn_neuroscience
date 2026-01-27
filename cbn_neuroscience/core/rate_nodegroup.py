# cbn_neuroscience/core/rate_nodegroup.py
import numpy as np
from cbn_neuroscience.core.neuron_model import NeuronModel

class RateNodeGroup(NeuronModel):
    """
    Modelo de tasa de población con funciones de activación flexibles.
    """
    def __init__(self, n_nodes, tau_A=20.0, dt=0.1, gain_function_type='sigmoid', **gain_params):
        super().__init__(n_nodes)
        self.tau_A = tau_A
        self.dt = dt
        self.gain_function_type = gain_function_type
        self.gain_params = gain_params

        self.A = np.zeros(n_nodes)

    def _sigmoid_gain_function(self, x):
        beta = self.gain_params.get('beta', 1.0)
        x0 = self.gain_params.get('x0', 5.0)
        return 1 / (1 + np.exp(-beta * (x - x0)))

    def _threshold_linear_gain_function(self, x):
        theta = self.gain_params.get('theta', 2.0)
        return np.maximum(0, x - theta)

    def _step_gain_function(self, x):
        # Emulado con una sigmoide de beta muy alto
        beta = 50.0
        x0 = self.gain_params.get('x0', 5.0)
        return 1 / (1 + np.exp(-beta * (x - x0)))

    def get_gain(self, x):
        if self.gain_function_type == 'sigmoid':
            return self._sigmoid_gain_function(x)
        elif self.gain_function_type == 'threshold_linear':
            return self._threshold_linear_gain_function(x)
        elif self.gain_function_type == 'step':
            return self._step_gain_function(x)
        return x # Ganancia lineal por defecto

    def update(self, **inputs):
        I_total = inputs.get('I_total', 0)
        target_A = self.get_gain(I_total)
        dA = (-self.A + target_A) / self.tau_A
        self.A += dA * self.dt
        self.A[self.A < 0] = 0

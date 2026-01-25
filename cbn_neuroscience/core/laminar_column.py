# cbn_neuroscience/core/laminar_column.py

import numpy as np
from cbnetwork.localnetwork import LocalNetwork

class LaminarColumn(LocalNetwork):
    """
    Representación de una columna cortical con estructura de 6 capas
    basada en Trappenberg Sección 1.2.7.
    """
    def __init__(self, index: int, internal_variables: list, layer_map: dict = None, **kwargs):
        # El constructor base espera 'index' e 'internal_variables'
        super().__init__(index, internal_variables)

        # Asignamos el layer_map después de llamar al constructor base
        self.layer_map = layer_map or self._default_layer_map(len(internal_variables))

        # Nivel 3 de Marr: Implementación (Vector de Sparse Coding)
        self.activity_history = []
        self.current_state = np.zeros(len(internal_variables))

    def _default_layer_map(self, n: int) -> dict:
        """Distribución aproximada de neuronas por capa."""
        # Necesitamos que los rangos se basen en los índices de las variables internas, no en 0..n-1
        start_index = self.internal_variables[0]
        end_index = self.internal_variables[-1]

        l23_end = start_index + int(n * 0.3)
        l4_end = l23_end + int(n * 0.2)

        return {
            'L2/3': list(range(start_index, l23_end)),      # Asociación / Feedback
            'L4':   list(range(l23_end, l4_end)),          # Input (Granular)
            'L5/6': list(range(l4_end, end_index + 1))       # Output / Deep (Infragranular)
        }

    def update_with_uncertainty(self, next_state: np.ndarray, p_error: float = 0.01) -> np.ndarray:
        """
        Implementa la Sección 1.5.3: El cerebro incierto (Stochastic Transitions)
        """
        # Introducimos ruido estocástico (Epistemological Uncertainty)
        noise_mask = np.random.random(next_state.shape) < p_error
        noisy_state = np.logical_xor(next_state, noise_mask).astype(int)

        self.current_state = noisy_state
        self.activity_history.append(self.current_state)

        return self.current_state

    def get_sparse_density(self) -> float:
        """
        Mide qué tan 'Sparse' es la activación actual (Sección 1.5.3)
        """
        return np.mean(self.current_state)

# cbn_neuroscience/core/anticipation_engine.py
import numpy as np
import numba

@numba.njit
def _normalized_hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la Distancia de Hamming Normalizada entre dos arrays de NumPy.
    Compilado con Numba para un alto rendimiento.
    """
    return np.sum(a != b) / len(a)

class AnticipationMonitor:
    """
    Un observador que monitorea el estado de una columna para detectar si ha convergido
    a un estado estable (atractor) utilizando la Distancia de Hamming.
    """
    def __init__(self, threshold: float = 0.05):
        self.history = []
        self.threshold = threshold

    def check_stability(self, state: np.ndarray) -> tuple[bool, str]:
        """
        Detecta si el sistema ha convergido a un concepto (atractor)
        comparando el estado actual con los estados anteriores.

        Args:
            state (np.ndarray): El estado actual de la red.

        Returns:
            tuple[bool, str]: Una tupla con un booleano que indica si se ha alcanzado la estabilidad
                              y un mensaje descriptivo.
        """
        for i, past_state in enumerate(self.history):
            distance = _normalized_hamming_distance(state, past_state)

            if distance < self.threshold:
                cycle_len = len(self.history) - i
                if cycle_len == 1:
                    return True, f"Convergencia a Punto Fijo (Distancia: {distance:.3f})"
                else:
                    return True, f"Convergencia a Ciclo Límite de longitud {cycle_len} (Distancia: {distance:.3f})"

        self.history.append(state.copy()) # Guardar una copia para evitar mutaciones
        return False, "Procesando..."

    def reset(self):
        """Reinicia el monitor para una nueva simulación."""
        self.history = []

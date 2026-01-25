# cbn_neuroscience/core/anticipation_engine.py

class AnticipationMonitor:
    """
    Un observador que monitorea el estado de una columna (o red) para
    detectar si ha caído en un estado estable (atractor).
    """
    def __init__(self):
        self.attractor_registry = set()
        self.history = []

    def check_stability(self, state: tuple) -> tuple[bool, str]:
        """
        Detecta si el sistema ha 'reconocido' el estímulo al caer en un atractor.

        Args:
            state (tuple): El estado actual de la red, convertido a una tupla para ser hashable.

        Returns:
            tuple[bool, str]: Una tupla con un booleano que indica si se ha alcanzado la estabilidad
                              y un mensaje descriptivo.
        """
        if state in self.attractor_registry:

            try:
                start_index = self.history.index(state)
                cycle_len = len(self.history) - start_index
                if cycle_len == 1:
                    return True, f"Concepto Reconocido (Punto Fijo: {state})"
                else:
                    return True, f"Concepto Reconocido (Ciclo Límite de longitud {cycle_len})"
            except ValueError:
                # Esto no debería ocurrir si el estado está en el registro.
                return True, "Concepto Reconocido (Atractor)"

        self.attractor_registry.add(state)
        self.history.append(state)
        return False, "Procesando..."

    def reset(self):
        """Reinicia el monitor para una nueva simulación."""
        self.attractor_registry.clear()
        self.history = []

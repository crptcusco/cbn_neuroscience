# cbn_neuroscience/core/compartmental_column.py

import numpy as np
from cbn_neuroscience.core.neuron_model import NeuronModel

class CompartmentalColumn:
    """
    Columna cortical genérica. Su única responsabilidad es mantener sus capas
    y actualizarlas con los inputs pre-calculados por el simulador.
    """
    def __init__(self, index: int, n_nodes_per_layer: dict, model_class: type[NeuronModel],
                 model_params: dict):
        self.index = index

        self.layers = {name: model_class(n_nodes=n_nodes, **model_params)
                       for name, n_nodes in n_nodes_per_layer.items()}

        self.is_spike_based = hasattr(next(iter(self.layers.values())), 'spikes')
        if self.is_spike_based:
            self.prev_spikes = {name: np.zeros(layer.n_nodes, dtype=bool) for name, layer in self.layers.items()}

    def update(self, step_time, layer_inputs: dict):
        """
        Args:
            step_time (float): Tiempo actual de la simulación.
            layer_inputs (dict): Inputs pre-calculados para cada capa.
        """
        # Primero, guardar los spikes actuales para el cálculo del siguiente paso
        if self.is_spike_based:
            for name, layer in self.layers.items():
                self.prev_spikes[name] = layer.spikes.copy()

        # Luego, actualizar cada capa con sus inputs
        for name, layer in self.layers.items():
            inputs_for_layer = layer_inputs.get(name, {})
            if self.is_spike_based:
                layer.update(step_time, **inputs_for_layer)
            else:
                layer.update(**inputs_for_layer)

    def get_state(self):
        """Devuelve el estado relevante (spikes o actividad) de cada capa."""
        if self.is_spike_based:
            return {name: layer.spikes for name, layer in self.layers.items()}
        else: # Rate-based
            return {name: layer.A for name, layer in self.layers.items()}

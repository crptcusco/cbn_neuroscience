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

    def update(self, layer_inputs: dict):
        """
        Args:
            layer_inputs (dict): Un diccionario donde cada clave es un nombre de capa
                                 y el valor es un dict de inputs para esa capa.
                                 Ej: {'L4': {'weighted_spikes': 0.5, 'I_noise': ...}}
        """
        for name, layer in self.layers.items():
            inputs_for_layer = layer_inputs.get(name, {})
            layer.update(**inputs_for_layer)

    def get_state(self):
        """Devuelve el estado relevante (spikes o actividad) de cada capa."""
        if self.is_spike_based:
            return {name: layer.spikes.copy() for name, layer in self.layers.items()}
        else: # Rate-based
            return {name: layer.A for name, layer in self.layers.items()}

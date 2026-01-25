# cbn_neuroscience/core/compartmental_column.py

import numpy as np
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup

class CompartmentalColumn:
    """
    Columna cortical con modelo GLIF, diseñada para interacciones complejas.
    """
    def __init__(self, index: int, n_nodes_per_layer: dict, coupling_rules: list, lif_params: dict = None):
        self.index = index
        self.coupling_rules = coupling_rules

        self.layers = {name: LIF_NodeGroup(n_nodes=n_nodes, **(lif_params or {}))
                       for name, n_nodes in n_nodes_per_layer.items()}

        self.prev_spikes = {name: np.zeros(layer.n_nodes, dtype=bool) for name, layer in self.layers.items()}

    def update(self, network_state: dict, ext_inputs: dict, noise_inputs: dict):
        """
        Args:
            network_state (dict): Spikes de la red en el paso anterior.
            ext_inputs (dict): Entradas externas para cada capa.
            noise_inputs (dict): Ruido para cada capa.
        """
        # --- 1. Calcular inputs totales para cada capa ---
        layer_inputs = {name: {'additive': np.zeros(layer.n_nodes), 'nmda': np.zeros(layer.n_nodes)}
                        for name, layer in self.layers.items()}

        # Añadir entradas externas
        for layer_name, inputs in ext_inputs.items():
            for input_type, value in inputs.items():
                if layer_name in layer_inputs:
                    layer_inputs[layer_name][input_type] += value

        # Procesar reglas de acoplamiento
        for rule in self.coupling_rules:
            if rule['target_col'] != self.index: continue

            source_spikes = [network_state[f'col_{c}'][l] for c, l in rule['sources']]

            # La interacción se basa en la tasa de disparo media de las fuentes
            mean_source_activity = [np.mean(s) for s in source_spikes]

            if rule['type'] == 'multiplicative':
                value = np.prod(mean_source_activity)
            else: # Additive
                value = np.sum(mean_source_activity)

            input_type = rule.get('synapse_type', 'additive')
            layer_inputs[rule['target_layer']][input_type] += rule['weight'] * value

        # --- 2. Actualizar cada capa ---
        for name, layer in self.layers.items():
            inputs = layer_inputs[name]
            layer.update(
                weighted_spikes=inputs['additive'],
                nmda_spikes=inputs['nmda'],
                I_noise=noise_inputs.get(name, 0)
            )

        # --- 3. Guardar estado de spikes para el siguiente ciclo ---
        for name, layer in self.layers.items():
            self.prev_spikes[name] = layer.spikes.copy()

    def get_layer_spikes(self):
        return {name: layer.spikes for name, layer in self.layers.items()}

    def get_layer_voltages(self):
        return {name: layer.v for name, layer in self.layers.items()}

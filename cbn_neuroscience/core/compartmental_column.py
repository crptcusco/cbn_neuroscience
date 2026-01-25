# cbn_neuroscience/core/compartmental_column.py

import numpy as np
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup

class CompartmentalColumn:
    """
    Columna cortical con modelo GLIF y sinapsis basadas en conductancia.
    El acoplamiento se basa en spikes, no en corrientes directas.
    """
    def __init__(self, index: int, n_nodes_per_layer: dict, g_axial: float = 0.5, lif_params: dict = None, **kwargs):
        self.index = index
        self.g_axial = g_axial

        if lif_params is None:
            lif_params = {}

        self.layers = {}
        for name, n_nodes in n_nodes_per_layer.items():
            self.layers[name] = LIF_NodeGroup(n_nodes=n_nodes, **lif_params)

        self.output_layer_name = 'L5/6'

        # Almacenar spikes del paso anterior para el acoplamiento
        self.prev_spikes = {name: np.zeros(layer.n_nodes, dtype=bool) for name, layer in self.layers.items()}

    def update(self, ext_spikes: np.ndarray, I_noise_total: np.ndarray, inter_column_spikes: np.ndarray):
        """
        Args:
            ext_spikes (np.ndarray): Pesos de spikes de estímulo externo a L4.
            I_noise_total (np.ndarray): Ruido de corriente para todas las capas.
            inter_column_spikes (np.ndarray): Pesos de spikes de otras columnas a L2/3.
        """
        # --- 1. Calcular los spikes de acoplamiento axial del paso ANTERIOR ---
        # El acoplamiento es un incremento de conductancia proporcional a g_axial
        # si la capa presináptica disparó en el paso anterior.

        # Asumimos conectividad total entre capas para simplificar (se promedia el efecto)
        spikes_4_to_23 = self.g_axial * np.mean(self.prev_spikes['L4'])
        spikes_23_to_56 = self.g_axial * np.mean(self.prev_spikes['L2/3'])
        spikes_23_to_4 = self.g_axial * np.mean(self.prev_spikes['L2/3'])
        spikes_56_to_23 = self.g_axial * np.mean(self.prev_spikes['L5/6'])

        # --- 2. Preparar los inputs (spikes pesados) para cada capa ---
        weighted_spikes = {name: np.zeros(layer.n_nodes) for name, layer in self.layers.items()}

        weighted_spikes['L4'] += ext_spikes + spikes_23_to_4
        weighted_spikes['L2/3'] += inter_column_spikes + spikes_4_to_23 + spikes_56_to_23
        weighted_spikes['L5/6'] += spikes_23_to_56

        # --- 3. Actualizar la dinámica GLIF para cada capa ---
        offset = 0
        for name, layer in self.layers.items():
            end = offset + layer.n_nodes
            noise_term = I_noise_total[offset:end]
            layer.update(weighted_spikes[name], noise_term)
            offset = end

        # --- 4. Guardar los spikes actuales para el siguiente paso ---
        for name, layer in self.layers.items():
            self.prev_spikes[name] = layer.spikes.copy()

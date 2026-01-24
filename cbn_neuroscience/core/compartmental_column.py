# cbn_neuroscience/core/compartmental_column.py

import numpy as np
from cbn_neuroscience.core.fhn_network import FHN_NodeGroup

class CompartmentalColumn:
    """
    Una columna cortical con un modelo de tres compartimentos (capas)
    que utiliza la dinámica FitzHugh-Nagumo y acoplamiento axial.
    """
    def __init__(self, index: int, n_nodes_per_layer: dict, g_axial: float = 0.5, **kwargs):
        self.index = index
        self.g_axial = g_axial

        total_nodes = sum(n_nodes_per_layer.values())
        self.internal_variables = list(range(index, index + total_nodes))

        # Crear un FHN_NodeGroup para cada capa (compartimento)
        self.layers = {}
        current_offset = 0
        for layer_name, n_nodes in n_nodes_per_layer.items():
            self.layers[layer_name] = FHN_NodeGroup(n_nodes=n_nodes)
            # Asignar índices globales a cada nodo de la capa para referencia externa
            self.layers[layer_name].global_indices = list(range(index + current_offset, index + current_offset + n_nodes))
            current_offset += n_nodes

        # El estado booleano de la columna lo determina la capa de salida
        self.output_layer_name = 'L5/6'
        self.current_state = np.zeros(total_nodes)

    def update(self, I_ext: np.ndarray, I_noise_total: np.ndarray = None):
        """
        Actualiza el estado de todos los compartimentos de la columna.
        """
        # --- 1. Calcular corrientes de acoplamiento interno (axial) ---
        v_l4_mean = np.mean(self.layers['L4'].v)
        v_l23_mean = np.mean(self.layers['L2/3'].v)
        v_l56_mean = np.mean(self.layers['L5/6'].v)

        # Acoplamiento Feedforward
        I_4_to_23 = self.g_axial * (v_l4_mean - v_l23_mean)
        I_23_to_56 = self.g_axial * (v_l23_mean - v_l56_mean)

        # Acoplamiento Feedback (Bidireccional)
        I_23_to_4 = self.g_axial * (v_l23_mean - v_l4_mean)
        I_56_to_23 = self.g_axial * (v_l56_mean - v_l23_mean)

        # --- 2. Sumar todas las corrientes y actualizar cada capa ---
        I_total = {name: np.zeros(layer.n_nodes) for name, layer in self.layers.items()}

        if I_ext is not None:
            I_total['L4'] += I_ext
        if I_noise_total is not None:
            offset = 0
            for name, layer in self.layers.items():
                end = offset + layer.n_nodes
                I_total[name] += I_noise_total[offset:end]
                offset = end

        # Sumar todas las corrientes de acoplamiento
        I_total['L4'] += I_23_to_4
        I_total['L2/3'] += I_4_to_23 + I_56_to_23
        I_total['L5/6'] += I_23_to_56

        # Actualizar la dinámica FHN para cada capa
        for name, layer in self.layers.items():
            layer.update(I_total[name])

        # --- 3. Actualizar el estado booleano de salida de la columna ---
        # El estado de salida de la columna completa lo determina la capa L5/6
        output_spikes = self.layers[self.output_layer_name].states

        # Mapear los spikes de la capa de salida al estado general de la columna
        output_indices_in_column = [self.internal_variables.index(i) for i in self.layers[self.output_layer_name].global_indices]
        self.current_state.fill(0)
        self.current_state[output_indices_in_column] = output_spikes

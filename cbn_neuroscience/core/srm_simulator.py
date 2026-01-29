# cbn_neuroscience/core/srm_simulator.py
import numpy as np
from collections import deque

class SRM_NetworkSimulator:
    """
    Orquesta la simulación de una red de columnas corticales basadas en el SRM.
    """
    def __init__(self, columns, edges, coupling_strength=1.0, delay_steps=1):
        self.columns = columns
        self.edges = edges
        self.coupling_strength = coupling_strength
        self.delay_steps = delay_steps

        # El buffer de retardo almacenará los spikes de salida de cada columna
        # para su entrega futura.
        self.delay_buffer = {i: deque([np.zeros(c.layers['L2/3'].n_nodes) for _ in range(delay_steps + 1)])
                             for i, c in enumerate(self.columns)}

    def run_step(self, external_stimuli, noise_stimuli):
        """
        Ejecuta un paso de la simulación de la red.

        Args:
            external_stimuli (dict): Estímulos externos para cada columna.
            noise_stimuli (dict): Ruido para cada columna.
        """
        # --- 1. Calcular los spikes de entrada inter-columna desde el buffer ---
        inter_column_inputs = {i: np.zeros_like(c.layers['L2/3'].n_nodes, dtype=float) for i, c in enumerate(self.columns)}

        for from_col_idx, to_col_idx in self.edges:
            # Recuperar los spikes que fueron emitidos 'delay_steps' atrás
            delayed_spikes = self.delay_buffer[from_col_idx].popleft()

            # Los spikes de la capa de salida (L5/6) de la columna 'from' se proyectan
            # a la capa de asociación (L2/3) de la columna 'to'.
            # Aquí asumimos una conectividad total para simplificar.
            if np.any(delayed_spikes):
                 inter_column_inputs[to_col_idx] += self.coupling_strength * np.mean(delayed_spikes)

        # --- 2. Actualizar cada columna con sus inputs ---
        for i, col in enumerate(self.columns):
            I_ext = external_stimuli.get(i, np.zeros(col.layers['L4'].n_nodes))
            I_noise = noise_stimuli.get(i, np.zeros(sum(l.n_nodes for l in col.layers.values())))
            I_inter = inter_column_inputs[i]

            col.update(I_ext, I_noise, I_inter)

        # --- 3. Almacenar los nuevos spikes de salida en el buffer ---
        for i, col in enumerate(self.columns):
            # Los spikes de salida son los de la capa L5/6
            output_spikes = col.layers[col.output_layer_name].spikes.astype(float)
            self.delay_buffer[i].append(output_spikes)

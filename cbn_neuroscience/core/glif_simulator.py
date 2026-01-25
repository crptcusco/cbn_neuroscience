# cbn_neuroscience/core/glif_simulator.py
import numpy as np
from collections import deque

class GLIFNetworkSimulator:
    """
    Orquesta la simulación de una red de columnas corticales basadas en el GLIF.
    """
    def __init__(self, columns, edges, coupling_strength=1.0, delay_ms=2.0, dt=0.1):
        self.columns = columns
        self.edges = edges
        self.coupling_strength = coupling_strength
        self.delay_steps = int(delay_ms / dt)

        # Buffer de retardo para los spikes de salida de cada columna
        self.delay_buffer = {i: deque([np.zeros(c.layers[c.output_layer_name].n_nodes, dtype=bool)
                                       for _ in range(self.delay_steps + 1)])
                             for i, c in enumerate(self.columns)}

    def run_step(self, external_stimuli, noise_stimuli):
        """
        Ejecuta un paso de la simulación de la red.

        Args:
            external_stimuli (dict): Pesos de spikes externos para cada columna.
            noise_stimuli (dict): Ruido de corriente para cada columna.
        """
        # --- 1. Calcular los spikes de entrada inter-columna desde el buffer ---
        inter_column_spikes = {i: np.zeros(c.layers['L2/3'].n_nodes) for i, c in enumerate(self.columns)}

        for from_col_idx, to_col_idx in self.edges:
            delayed_spikes = self.delay_buffer[from_col_idx].popleft()

            # El promedio de spikes de la capa de salida (L5/6) se convierte en un
            # incremento de conductancia en la capa de asociación (L2/3) de la columna destino.
            if np.any(delayed_spikes):
                 inter_column_spikes[to_col_idx] += self.coupling_strength * np.mean(delayed_spikes)

        # --- 2. Actualizar cada columna ---
        for i, col in enumerate(self.columns):
            total_nodes = sum(l.n_nodes for l in col.layers.values())

            ext_spikes = external_stimuli.get(i, np.zeros(col.layers['L4'].n_nodes))
            I_noise = noise_stimuli.get(i, np.zeros(total_nodes))
            inter_spikes = inter_column_spikes[i]

            col.update(ext_spikes, I_noise, inter_spikes)

        # --- 3. Almacenar los nuevos spikes de salida en el buffer ---
        for i, col in enumerate(self.columns):
            output_spikes = col.layers[col.output_layer_name].spikes
            self.delay_buffer[i].append(output_spikes.copy())

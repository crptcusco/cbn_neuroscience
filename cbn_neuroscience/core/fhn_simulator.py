# cbn_neuroscience/core/fhn_simulator.py

import numpy as np
from collections import deque, defaultdict

class FHNNetworkSimulator:
    """
    Orquesta la simulación de una red de columnas compartimentales con una topología
    definida por el usuario y retardo sináptico.
    """
    def __init__(self, columns: list, edges: list, coupling_strength: float = 0.5, delay_steps: int = 2):
        self.columns_map = {col.index: col for col in columns}
        self.edges = edges
        self.coupling_strength = coupling_strength
        self.delay_steps = delay_steps

        self.coupling_buffers = [deque([0.0] * delay_steps) for _ in self.edges]

        from .compartmental_column import CompartmentalColumn
        if not all(isinstance(col, CompartmentalColumn) for col in self.columns_map.values()):
            raise TypeError("Este simulador solo opera con una lista de CompartmentalColumn.")

    def run_step(self, external_stimuli: dict = None, noise_stimuli: dict = None):
        """
        Ejecuta un único paso de tiempo para toda la red.
        """
        if external_stimuli is None: external_stimuli = {}
        if noise_stimuli is None: noise_stimuli = {}

        # 1. Leer señales retardadas y acumularlas por columna destino
        delayed_coupling_signals = defaultdict(float)
        for i, edge in enumerate(self.edges):
            delayed_signal = self.coupling_buffers[i].popleft()
            target_col_index = edge[1]
            delayed_coupling_signals[target_col_index] += delayed_signal

        # 2. Actualizar cada columna
        for index, col in self.columns_map.items():
            I_ext = external_stimuli.get(index, np.zeros(col.layers['L4'].n_nodes))
            I_noise = noise_stimuli.get(index, np.zeros(sum(l.n_nodes for l in col.layers.values())))

            I_ext += delayed_coupling_signals[index]

            col.update(I_ext=I_ext, I_noise_total=I_noise)

        # 3. Calcular nuevas señales y añadirlas a los búferes
        for i, edge in enumerate(self.edges):
            source_col = self.columns_map[edge[0]]
            source_output_spikes = source_col.layers['L5/6'].states
            new_signal = np.sum(source_output_spikes) * self.coupling_strength
            self.coupling_buffers[i].append(new_signal)

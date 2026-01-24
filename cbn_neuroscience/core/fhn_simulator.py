# cbn_neuroscience/core/fhn_simulator.py

import numpy as np
from collections import deque

class FHNNetworkSimulator:
    """
    Orquesta la simulación de una red en cadena de columnas compartimentales
    con retardo sináptico.
    """
    def __init__(self, columns: list, coupling_strength: float = 0.5, delay_steps: int = 2):
        self.columns = columns
        self.coupling_strength = coupling_strength
        self.delay_steps = delay_steps

        # Crear un búfer de retardo para cada conexión
        self.coupling_buffers = [deque([0.0] * delay_steps) for _ in range(len(self.columns) - 1)]

        from .compartmental_column import CompartmentalColumn
        if not all(isinstance(col, CompartmentalColumn) for col in self.columns):
            raise TypeError("Este simulador solo opera con una lista de CompartmentalColumn.")

    def run_step(self, external_stimuli: dict = None, noise_stimuli: dict = None):
        """
        Ejecuta un único paso de tiempo para toda la red, aplicando el retardo sináptico.
        """
        if external_stimuli is None: external_stimuli = {}
        if noise_stimuli is None: noise_stimuli = {}

        # 1. Leer las señales de acoplamiento retardadas de los búferes
        delayed_coupling_signals = {}
        for i in range(len(self.columns) - 1):
            delayed_signal = self.coupling_buffers[i].popleft()
            target_col_index = self.columns[i+1].index
            delayed_coupling_signals[target_col_index] = delayed_signal

        # 2. Actualizar cada columna con las corrientes apropiadas
        for col in self.columns:
            I_ext = external_stimuli.get(col.index, np.zeros(col.layers['L4'].n_nodes))
            I_noise = noise_stimuli.get(col.index, np.zeros(sum(l.n_nodes for l in col.layers.values())))

            # La corriente de acoplamiento (ya retardada) se suma a la corriente externa
            I_ext += delayed_coupling_signals.get(col.index, 0)

            col.update(I_ext=I_ext, I_noise_total=I_noise)

        # 3. Calcular las nuevas señales de acoplamiento y añadirlas al búfer para el futuro
        for i in range(len(self.columns) - 1):
            source_col = self.columns[i]
            source_output_spikes = source_col.layers['L5/6'].states
            new_signal = np.sum(source_output_spikes) * self.coupling_strength
            self.coupling_buffers[i].append(new_signal)

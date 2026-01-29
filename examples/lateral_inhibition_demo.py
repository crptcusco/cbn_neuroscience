# examples/lateral_inhibition_demo.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator
from cbn_neuroscience.core.plasticity_manager import PlasticityManager

# --- 1. Parámetros ---
DT = 0.1
SIM_TIME_MS = 3000 # Simulación más larga para que los pesos converjan
N_STEPS = int(SIM_TIME_MS / DT)
N_COLUMNS = 2
N_NEURONS = 20

lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
              'R_m': 10.0, 'tau_syn_exc': 5.0, 'tau_syn_inh': 10.0, 'delta': 2.0, 'dt': DT}
stdp_params = {'a_plus': 0.05, 'a_minus': -0.05, 'tau_plus': 20.0, 'tau_minus': 20.0,
               'w_max': 1.0, 'w_min': -1.0}

# --- 2. Construcción de la Red ---
# Conexiones recíprocas entre las dos columnas
rules = [
    {'sources': [(0, 'L5')], 'target_col': 1, 'target_layer': 'L5', 'type': 'additive', 'weight': 0.5},
    {'sources': [(1, 'L5')], 'target_col': 0, 'target_layer': 'L5', 'type': 'additive', 'weight': 0.5}
]

columns = [CompartmentalColumn(index=i, n_nodes_per_layer={'L5': N_NEURONS},
                               model_class=LIF_NodeGroup, model_params=lif_params)
           for i in range(N_COLUMNS)]

plasticity = PlasticityManager(rule_type='stdp_multiplicative', **stdp_params)
simulator = NetworkSimulator(columns, rules, plasticity)

# --- 3. Bucle de Simulación ---
# Estímulo idéntico y correlacionado para ambas columnas
stimulus_strength = 1.8
noise_sigma = 1.5

# Historial de pesos
weight_history = {
    '0_to_1': np.zeros(N_STEPS),
    '1_to_0': np.zeros(N_STEPS)
}

print("Ejecutando simulación de inhibición lateral emergente...")
for step in range(N_STEPS):
    # El mismo input para ambas para crear competencia
    ext_inputs = {
        0: {'L5': {'exc_spikes': stimulus_strength, 'I_noise': np.random.normal(0, noise_sigma, N_NEURONS)}},
        1: {'L5': {'exc_spikes': stimulus_strength, 'I_noise': np.random.normal(0, noise_sigma, N_NEURONS)}}
    }

    simulator.run_step(step, ext_inputs)

    # Guardar pesos
    weight_history['0_to_1'][step] = simulator.connection_manager.get_weight(0, 'L5', 1, 'L5')
    weight_history['1_to_0'][step] = simulator.connection_manager.get_weight(1, 'L5', 0, 'L5')

    if step % 1000 == 0:
        print(f"  Paso {step}/{N_STEPS}...")

print("Simulación completada.")
# (La visualización y la lógica final de plasticidad se añadirán después)

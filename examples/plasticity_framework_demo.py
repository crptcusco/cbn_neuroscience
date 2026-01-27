# examples/plasticity_framework_demo.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

# --- 1. Definición de la Red y Reglas ---
rate_params = {'tau_A': 20.0, 'gain_function_type': 'sigmoid', 'beta': 1.0, 'x0': 5.0}
n_nodes_per_layer = {'L4': 1, 'L5': 1}

# Regla: Conexión simple de L4 a L5 en la misma columna
rules = [{'sources': [(0, 'L4')], 'target_col': 0, 'target_layer': 'L5', 'type': 'additive', 'weight': 0.5}]

columns = [CompartmentalColumn(index=0, n_nodes_per_layer=n_nodes_per_layer,
                             model_class=RateNodeGroup, model_params=rate_params)]

simulator = NetworkSimulator(columns, coupling_rules=rules)

# --- 2. Definición de la Regla de Plasticidad "Dummy" ---
def simple_potentiation_rule(current_weight, pre_activities, post_activity):
    """Una regla simple que incrementa el peso si hay actividad."""
    # pre_activities es una lista, tomamos el primero
    pre_activity = pre_activities[0]
    if pre_activity > 0.1 and post_activity > 0.1:
        return current_weight + 0.01 # Potenciación a largo plazo (LTP) simple
    return current_weight # Sin cambios

# --- 3. Simulación ---
DT = 0.1
SIM_TIME_MS = 200
N_STEPS = int(SIM_TIME_MS / DT)

print("Ejecutando demostración del framework de plasticidad...")
for step in range(N_STEPS):
    # Aplicar un estímulo constante para inducir actividad
    ext_inputs = {0: {'L4': {'I_total': 10.0}}}

    simulator.run_step(ext_inputs)
    simulator.apply_plasticity(simple_potentiation_rule)
    simulator.record_weights() # Guardar los pesos en cada paso

print("Simulación completada.")

# --- 4. Visualización de la Evolución del Peso ---
history = np.array(simulator.connection_manager.weight_history)

# Extraer el peso de la conexión de L4 a L5
source_idx = simulator.connection_manager.layer_map[(0, 'L4')]
target_idx = simulator.connection_manager.layer_map[(0, 'L5')]
weight_evolution = history[:, target_idx, source_idx]

time_axis = np.arange(len(weight_evolution)) * DT

plt.figure(figsize=(10, 6))
plt.plot(time_axis, weight_evolution, label='Peso de la sinapsis L4 -> L5')
plt.title('Evolución de un Peso Sináptico con una Regla de Plasticidad')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Peso Sináptico (w)')
plt.legend()
plt.grid(True)
plt.savefig('weight_evolution.png')

print("Gráfica de evolución de pesos guardada en 'weight_evolution.png'")

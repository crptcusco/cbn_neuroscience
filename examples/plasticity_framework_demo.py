# examples/plasticity_framework_demo.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

from cbn_neuroscience.core.plasticity_manager import PlasticityManager

# --- 1. Definición de la Red y Reglas ---
rate_params = {'tau_A': 20.0, 'gain_function_type': 'sigmoid', 'beta': 1.0, 'x0': 5.0}
n_nodes_per_layer = {'L4': 1, 'L5': 1}

# Regla: Conexión simple de L4 a L5 en la misma columna
rules = [{'sources': [(0, 'L4')], 'target_col': 0, 'target_layer': 'L5', 'type': 'additive', 'weight': 0.5}]

columns = [CompartmentalColumn(index=0, n_nodes_per_layer=n_nodes_per_layer,
                             model_class=RateNodeGroup, model_params=rate_params)]

# --- 2. Configuración del PlasticityManager ---
# Usaremos la regla de covarianza para simular una potenciación simple.
# Si la actividad pre y post está por encima de la media, el peso crecerá.
cov_params = {'learning_rate': 0.05, 'w_max': 1.0, 'w_min': 0.0}
plasticity = PlasticityManager(rule_type='covariance', **cov_params)

# El NetworkSimulator ahora maneja la plasticidad internamente
simulator = NetworkSimulator(columns, rules, plasticity_manager=plasticity)

# --- 3. Simulación ---
DT = 0.1
SIM_TIME_MS = 2000 # Simulación más larga para ver el cambio
N_STEPS = int(SIM_TIME_MS / DT)
weight_history = np.zeros(N_STEPS)

print("Ejecutando demostración del framework de plasticidad...")
for step in range(N_STEPS):
    # Aplicar un estímulo constante para inducir actividad
    ext_inputs = {0: {'L4': {'I_noise': 10.0}}}

    simulator.run_step(step, ext_inputs)

    # Registrar el peso actual
    current_weight = simulator.connection_manager.get_weight(0, 'L4', 0, 'L5')
    weight_history[step] = current_weight

print("Simulación completada.")

# --- 4. Visualización de la Evolución del Peso ---
time_axis = np.arange(N_STEPS) * DT

plt.figure(figsize=(10, 6))
plt.plot(time_axis, weight_history, label='Peso de la sinapsis L4 -> L5')
plt.title('Evolución de un Peso Sináptico con una Regla de Plasticidad')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Peso Sináptico (w)')
plt.legend()
plt.grid(True)
plt.savefig('weight_evolution.png')

print("Gráfica de evolución de pesos guardada en 'weight_evolution.png'")

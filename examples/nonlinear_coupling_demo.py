# examples/nonlinear_coupling_demo.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

# --- 1. Definición de la Red ---
# Usaremos un modelo de tasa para simplificar la demostración
rate_params = {'tau_A': 10.0, 'gain_function_type': 'sigmoid', 'beta': 4.0, 'x0': 0.5}

# Col 0: Entrada de señal
# Col 1: Entrada de compuerta (gate)
# Col 2: Salida, recibe ambas señales
n_nodes_per_layer = {'L5': 1}
columns = [
    CompartmentalColumn(index=0, n_nodes_per_layer=n_nodes_per_layer, model_class=RateNodeGroup, model_params=rate_params),
    CompartmentalColumn(index=1, n_nodes_per_layer=n_nodes_per_layer, model_class=RateNodeGroup, model_params=rate_params),
    CompartmentalColumn(index=2, n_nodes_per_layer=n_nodes_per_layer, model_class=RateNodeGroup, model_params=rate_params)
]

# Regla: La salida de la Col 2 es el producto de la actividad de Col 0 y Col 1
rules = [
    {'sources': [(0, 'L5'), (1, 'L5')], 'target_col': 2, 'target_layer': 'L5',
     'type': 'multiplicative', 'weight': 1.0}
]

simulator = NetworkSimulator(columns, rules)

# --- 2. Simulación ---
DT = 0.1
SIM_TIME_MS = 250
N_STEPS = int(SIM_TIME_MS / DT)

history = {
    'signal_in': np.zeros(N_STEPS),
    'gate_in': np.zeros(N_STEPS),
    'output': np.zeros(N_STEPS)
}

print("Ejecutando simulación de acoplamiento no lineal (Compuerta AND)...")
for step in range(N_STEPS):
    time_ms = step * DT

    # La señal de entrada (Col 0) está siempre activa
    signal_input = 1.0

    # La compuerta (Col 1) se activa solo en la segunda mitad de la simulación
    gate_input = 1.0 if time_ms > SIM_TIME_MS / 2 else 0.0

    ext_inputs = {
        0: {'L5': {'I_noise': signal_input}},
        1: {'L5': {'I_noise': gate_input}}
    }

    simulator.run_step(step, ext_inputs)

    # Guardar historial
    history['signal_in'][step] = columns[0].layers['L5'].A[0]
    history['gate_in'][step] = columns[1].layers['L5'].A[0]
    history['output'][step] = columns[2].layers['L5'].A[0]

print("Simulación completada.")

# --- 3. Visualización ---
time_axis = np.arange(N_STEPS) * DT
plt.figure(figsize=(12, 7))

plt.plot(time_axis, history['signal_in'], 'b-', label='Entrada de Señal (Col 0)')
plt.plot(time_axis, history['gate_in'], 'g--', label='Entrada de Compuerta (Col 1)')
plt.plot(time_axis, history['output'], 'r-', linewidth=3, label='Salida (Col 2)')

plt.title('Demostración de Acoplamiento No Lineal Multiplicativo (Compuerta AND)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Actividad de la Población')
plt.axvline(SIM_TIME_MS / 2, color='k', linestyle=':', label='Compuerta Activada')
plt.grid(True)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.savefig('nonlinear_coupling_demo.png')

print("Gráfica guardada en 'nonlinear_coupling_demo.png'")

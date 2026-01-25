# examples/nmda_gate_demo.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

# --- 1. Definición de la Red y Reglas ---
lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
              'R_m': 10.0, 'tau_syn_exc': 5.0, 'delta': 2.0, 'dt': 0.1}
n_nodes = 10

# Regla: Conexión NMDA de Col 1 a Col 0
rule_nmda = {'sources': [(1, 'L5')], 'target_col': 0, 'target_layer': 'L5',
             'type': 'additive', 'synapse_type': 'nmda', 'weight': 0.8}

columns = [
    CompartmentalColumn(index=0, n_nodes_per_layer={'L5': n_nodes}, lif_params=lif_params, coupling_rules=[rule_nmda]),
    CompartmentalColumn(index=1, n_nodes_per_layer={'L5': n_nodes}, lif_params=lif_params, coupling_rules=[])
]

# --- 2. Simulación ---
DT = 0.1; SIM_TIME_MS = 200; N_STEPS = int(SIM_TIME_MS / DT)
history = {'col_0_v': np.zeros(N_STEPS), 'col_1_spikes': np.zeros(N_STEPS)}
network_state = {'col_0': {'L5': np.zeros(n_nodes, dtype=bool)}, 'col_1': {'L5': np.zeros(n_nodes, dtype=bool)}}

print("Ejecutando simulación de compuerta NMDA...")
for step in range(N_STEPS):
    ext_inputs = [{}, {}]
    noise_inputs = [{'L5': np.random.normal(0, 0.5, n_nodes)}, {'L5': np.random.normal(0, 0.5, n_nodes)}]

    # Escenario 1 (50-100ms): Col 1 dispara, Col 0 en reposo
    if 50 / DT < step < 100 / DT:
        ext_inputs[1] = {'L5': {'additive': 2.0}}

    # Escenario 2 (150-200ms): Col 0 se despolariza, LUEGO Col 1 dispara
    if 150 / DT < step:
        ext_inputs[0] = {'L5': {'additive': 1.2}} # Input de preparación a Col 0
        ext_inputs[1] = {'L5': {'additive': 2.0}} # Input de disparo a Col 1

    # Actualizar Red
    for i, col in enumerate(columns):
        col.update(network_state=network_state, ext_inputs=ext_inputs[i], noise_inputs=noise_inputs[i])

    # Guardar estado de spikes del paso actual para el siguiente ciclo
    for i, col in enumerate(columns):
        network_state[f'col_{i}'] = col.get_layer_spikes()

    # Guardar historial para la gráfica
    history['col_0_v'][step] = np.mean(columns[0].get_layer_voltages()['L5'])
    history['col_1_spikes'][step] = np.any(columns[1].get_layer_spikes()['L5'])

# --- 3. Visualización ---
time_axis = np.arange(N_STEPS) * DT
fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.set_xlabel('Tiempo (ms)')
ax1.set_ylabel('Voltaje Promedio Col 0 (mV)', color='b')
ax1.plot(time_axis, history['col_0_v'], 'b-', label='Voltaje Col 0')
ax1.axhline(-60, color='gray', linestyle=':', label='Umbral NMDA (-60mV)')
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Disparo Col 1', color='r')
spike_times = time_axis[history['col_1_spikes'].astype(bool)]
ax2.plot(spike_times, np.ones_like(spike_times), 'r.', markersize=8, label='Spikes Col 1')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_ylim(0, 1.2)

plt.title('Demostración de Compuerta NMDA Dependiente de Voltaje')
fig.tight_layout()
plt.savefig('nmda_gate_demo.png')
print("Simulación completada. Gráfica guardada en 'nmda_gate_demo.png'")

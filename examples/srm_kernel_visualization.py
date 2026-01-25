# examples/srm_kernel_visualization.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.srm_nodegroup import SRM_NodeGroup

# --- 1. Parámetros del Modelo y Simulación ---
DT = 0.1  # ms
SIMULATION_TIME_MS = 150  # ms
N_STEPS = int(SIMULATION_TIME_MS / DT)

SRM_PARAMS = {
    'tau_m': 15.0,
    'theta': -55.0,
    'v_rest': -70.0,
    'dt': DT
}
N_NEURONS = 1

# --- 2. Creación del Tren de Pulsos de Entrada ---
# Spikes de entrada en momentos específicos para demostrar la integración
input_spike_times_ms = [20, 40, 60, 100, 105, 110]
input_spike_steps = [int(t / DT) for t in input_spike_times_ms]
input_spikes = np.zeros(N_STEPS)
input_spikes[input_spike_steps] = 1.0

# Peso sináptico de la conexión de entrada
W_IN = 8.0

# --- 3. Inicialización del Modelo y Simulación ---
srm_group = SRM_NodeGroup(n_nodes=N_NEURONS, **SRM_PARAMS)

# Contenedores para registrar la historia
v_history = np.zeros(N_STEPS)
h_syn_history = np.zeros(N_STEPS)
h_ref_history = np.zeros(N_STEPS)
spike_history = np.zeros(N_STEPS, dtype=bool)

for i in range(N_STEPS):
    weighted_input = W_IN * input_spikes[i]
    srm_group.update(np.array([weighted_input]))

    v_history[i] = srm_group.v[0]
    h_syn_history[i] = srm_group.h_syn[0]
    h_ref_history[i] = srm_group.h_ref[0]
    spike_history[i] = srm_group.spikes[0]

# --- 4. Generación de la Gráfica ---
print("Generando la gráfica de visualización de kernels SRM...")
fig, ax = plt.subplots(figsize=(15, 8))
time_axis = np.arange(N_STEPS) * DT

# Dibujar las contribuciones de los kernels
ax.plot(time_axis, SRM_PARAMS['v_rest'] + h_syn_history, label='Potencial Sináptico (Suma de ε-kernels)', linestyle='--', color='green')
ax.plot(time_axis, SRM_PARAMS['v_rest'] + h_ref_history, label='Potencial Refractario (Suma de η-kernels)', linestyle='--', color='orange')

# Dibujar el potencial total de membrana
ax.plot(time_axis, v_history, label='Potencial de Membrana Total (v)', color='blue', linewidth=2)

# Dibujar el umbral y los spikes
ax.axhline(SRM_PARAMS['theta'], color='r', linestyle=':', label=f'Umbral (θ={SRM_PARAMS["theta"]} mV)')
output_spike_times = time_axis[spike_history]
ax.plot(output_spike_times, np.full_like(output_spike_times, SRM_PARAMS['theta']), 'rx', markersize=12, markeredgewidth=2, label='Spike de Salida')

# Marcar los spikes de entrada para referencia
ax.stem(input_spike_times_ms, np.full(len(input_spike_times_ms), -50), linefmt='k:', markerfmt='ko', basefmt=" ", label='Spikes de Entrada (pesados)')


ax.set_title('Visualización de Kernels en el Spike-Response Model (SRM)')
ax.set_xlabel('Tiempo (ms)')
ax.set_ylabel('Potencial de Membrana (mV)')
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_ylim(-75, -50)

plt.tight_layout()
plt.savefig('srm_kernels.png')
print("Gráfica guardada en 'srm_kernels.png'")

# examples/stdp_self_synchronization.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

# --- 1. Parámetros ---
DT = 0.1
SIM_TIME_MS = 2000
N_STEPS = int(SIM_TIME_MS / DT)
N_COLUMNS = 4
N_NEURONS_PER_LAYER = 10

lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
              'R_m': 10.0, 'tau_syn_exc': 5.0, 'delta': 2.0, 'dt': DT}
stdp_params = {'tau_plus': 20.0, 'tau_minus': 20.0, 'A_plus': 0.05, 'A_minus': -0.05,
               'max_weight': 2.0, 'calcium_threshold': -60.0}

from cbn_neuroscience.core.plasticity_manager import PlasticityManager
from scipy.ndimage import gaussian_filter1d

# --- 2. Construcción de la Red ---
# Conectividad total entre las capas de salida (L5) de todas las columnas
rules = []
for i in range(N_COLUMNS):
    for j in range(N_COLUMNS):
        if i == j: continue # No hay autoconexiones
        rules.append({'sources': [(i, 'L5')], 'target_col': j, 'target_layer': 'L5',
                      'type': 'additive', 'weight': 0.5}) # Peso inicial uniforme

columns = [CompartmentalColumn(index=i, n_nodes_per_layer={'L5': N_NEURONS_PER_LAYER},
                               model_class=LIF_NodeGroup, model_params=lif_params)
           for i in range(N_COLUMNS)]

plasticity = PlasticityManager(rule_type='stdp_multiplicative', **stdp_params)
simulator = NetworkSimulator(columns, rules, plasticity_manager=plasticity)

# --- 3. Bucle de Simulación ---
# Estímulo oscilatorio a 42 Hz con fases aleatorias
freq = 42.0 # Hz
phases = np.random.rand(N_COLUMNS) * 2 * np.pi
time_axis = np.arange(N_STEPS) * DT # en milisegundos

# Historial de spikes y pesos
spike_history = {f'col_{i}': np.zeros((N_STEPS, N_NEURONS_PER_LAYER), dtype=bool) for i in range(N_COLUMNS)}
initial_weights = simulator.connection_manager.weights.copy()

print("Ejecutando simulación de auto-sincronización con STDP...")
for step in range(N_STEPS):
    ext_inputs = {}
    for i in range(N_COLUMNS):
        # Input oscilatorio + ruido
        oscillation = 0.8 * (1 + np.sin(2 * np.pi * freq * (time_axis[step] / 1000.0) + phases[i]))
        noise = np.random.normal(0, 0.5, N_NEURONS_PER_LAYER)
        ext_inputs[i] = {'L5': {'exc_spikes': oscillation, 'I_noise': noise}}

    simulator.run_step(step, ext_inputs)

    # Guardar spikes
    for i in range(N_COLUMNS):
        spike_history[f'col_{i}'][step, :] = columns[i].layers['L5'].spikes

final_weights = simulator.connection_manager.weights.copy()
print("Simulación completada.")


# --- 4. Visualización de Resultados ---
print("Generando gráficas de resultados...")

# Gráfica 1: Raster Plot
fig, axes = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

# Raster Plot
colors = ['blue', 'green', 'red', 'purple']
all_spike_times = []
all_neuron_ids = []
for i in range(N_COLUMNS):
    for neuron_idx in range(N_NEURONS_PER_LAYER):
        spike_times = time_axis[spike_history[f'col_{i}'][:, neuron_idx]]
        all_spike_times.extend(spike_times)
        all_neuron_ids.extend([neuron_idx + i * N_NEURONS_PER_LAYER] * len(spike_times))

axes[0].scatter(all_spike_times, all_neuron_ids, c=[colors[idx // N_NEURONS_PER_LAYER] for idx in all_neuron_ids], marker='.', s=10)
axes[0].set_title('Raster Plot de Sincronización de Spikes con STDP')
axes[0].set_xlabel('Tiempo (ms)')
axes[0].set_ylabel('ID de Neurona')
axes[0].grid(True, linestyle=':', alpha=0.5)

# Gráfica 2: Actividad de Población
for i in range(N_COLUMNS):
    activity = np.mean(spike_history[f'col_{i}'], axis=1)
    smoothed_activity = gaussian_filter1d(activity, sigma=20)
    axes[1].plot(time_axis, smoothed_activity, color=colors[i], label=f'Actividad Col {i}')
axes[1].set_title('Actividad de Población Suavizada')
axes[1].set_xlabel('Tiempo (ms)')
axes[1].set_ylabel('Actividad A(t)')
axes[1].legend()

plt.tight_layout()
plt.savefig('stdp_raster_plot.png')

# Gráfica 3: Matrices de Pesos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vmax = np.max([initial_weights, final_weights])

im1 = axes[0].imshow(initial_weights, cmap='viridis', interpolation='none', vmin=0, vmax=vmax)
axes[0].set_title('Matriz de Pesos Inicial')
axes[0].set_xlabel('Capa Presináptica')
axes[0].set_ylabel('Capa Postsináptica')

im2 = axes[1].imshow(final_weights, cmap='viridis', interpolation='none', vmin=0, vmax=vmax)
axes[1].set_title('Matriz de Pesos Final (Después de STDP)')
axes[1].set_xlabel('Capa Presináptica')

fig.colorbar(im2, ax=axes.ravel().tolist())
plt.tight_layout()
plt.savefig('stdp_weight_matrices.png')

print("Gráficas guardadas.")

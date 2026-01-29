# examples/synaptic_competition_demo.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator
from cbn_neuroscience.core.plasticity_manager import PlasticityManager

# --- 1. Parámetros ---
DT = 0.1
SIM_TIME_MS = 5000 # Simulación larga para observar la estabilización
N_STEPS = int(SIM_TIME_MS / DT)
N_COLUMNS = 4
N_NEURONS = 10

lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
              'R_m': 10.0, 'tau_syn_exc': 5.0, 'tau_syn_inh': 5.0, 'delta': 2.0, 'dt': DT}

# --- Tarea 1: Parámetros STDP con LTD dominante ---
# Area LTP = A+ * tau+
# Area LTD = A- * tau-
# Queremos que |Area LTD| sea ~1.1 * |Area LTP|
A_plus = 0.1
tau_plus = 20.0
tau_minus = 20.0
# A_minus * tau_minus = -1.1 * A_plus * tau_plus
A_minus = -1.1 * (A_plus * tau_plus) / tau_minus

stdp_params = {'a_plus': A_plus, 'a_minus': A_minus, 'tau_plus': tau_plus, 'tau_minus': tau_minus,
               'w_max': 1.0, 'w_min': 0.0} # Empezamos sin pesos negativos

# --- 2. Construcción de la Red ---
rules = []
for i in range(N_COLUMNS):
    for j in range(N_COLUMNS):
        if i == j: continue
        rules.append({'sources': [(i, 'L5')], 'target_col': j, 'target_layer': 'L5',
                      'type': 'additive', 'weight': np.random.rand() * 0.5}) # Pesos iniciales aleatorios

columns = [CompartmentalColumn(index=i, n_nodes_per_layer={'L5': N_NEURONS},
                               model_class=LIF_NodeGroup, model_params=lif_params)
           for i in range(N_COLUMNS)]

plasticity = PlasticityManager(rule_type='stdp_multiplicative', **stdp_params)
simulator = NetworkSimulator(columns, rules, plasticity)

# --- 3. Bucle de Simulación ---
stim_probability = 0.1
stim_strength = 2.0
noise_sigma = 1.0

# Monitores
total_spikes_history = np.zeros(N_STEPS)
spike_train_col1 = []
spike_train_col2 = []

print("Ejecutando simulación con LTD dominante...")
for step in range(N_STEPS):
    ext_inputs = {}
    for i in range(N_COLUMNS):
        noise = np.random.normal(0, noise_sigma, N_NEURONS)
        stim = stim_strength if np.random.rand() < stim_probability else 0
        ext_inputs[i] = {'L5': {'exc_spikes': stim, 'I_noise': noise}}

    simulator.run_step(step, ext_inputs)
    simulator.apply_plasticity(step * DT)

    # Registrar la tasa de disparo total y los trenes de picos
    total_spikes = 0
    for i in range(N_COLUMNS):
        current_spikes = np.sum(simulator.columns[i].layers['L5'].spikes)
        total_spikes += current_spikes
        if i == 0: # Columna 1
            spike_train_col1.append(current_spikes)
        elif i == 1: # Columna 2
            spike_train_col2.append(current_spikes)

    total_spikes_history[step] = total_spikes

print("Simulación completada.")

# --- 4. Análisis de Estabilidad ---
# Suavizar la tasa de disparo para ver la tendencia
time_axis = np.arange(N_STEPS) * DT
activity_rate = total_spikes_history / (N_COLUMNS * N_NEURONS)
smoothed_rate = np.convolve(activity_rate, np.ones(1000)/1000, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(time_axis[len(time_axis)-len(smoothed_rate):], smoothed_rate)
plt.title('Tasa de Disparo Promedio de la Red (con LTD Dominante)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Tasa de Disparo Normalizada')
plt.grid(True)
plt.savefig('firing_rate_stability.png')
print("\nGráfica de estabilidad de tasa de disparo guardada.")

# --- 5. Análisis de Distribución de Pesos (Tarea 2) ---
final_weights = simulator.connection_manager.weights
# Aplanar la matriz para el histograma, ignorando los ceros (conexiones no existentes)
flat_weights = final_weights[final_weights > 0].flatten()

plt.figure(figsize=(10, 6))
plt.hist(flat_weights, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribución Final de Pesos Sinápticos')
plt.xlabel('Fuerza Sináptica (w)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.savefig('weight_distribution.png')
print("Histograma de distribución de pesos guardado en 'weight_distribution.png'.")

# --- 6. Análisis de Correlación Cruzada (Tarea 3) ---
# Convertir listas a arrays de numpy para el cálculo
spike_train_col1 = np.array(spike_train_col1)
spike_train_col2 = np.array(spike_train_col2)

# Normalizar las señales para que la correlación no dependa de la tasa de disparo absoluta
def z_score_normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

norm_col1 = z_score_normalize(spike_train_col1)
norm_col2 = z_score_normalize(spike_train_col2)

# Calcular la correlación cruzada
correlation = np.correlate(norm_col1, norm_col2, mode='full')
lags = np.arange(-len(norm_col2) + 1, len(norm_col1))

# Limitar los lags a un rango más interpretable (ej. +/- 200 ms)
lag_limit_ms = 200
lag_limit_steps = int(lag_limit_ms / DT)
center_idx = len(correlation) // 2
start_idx = center_idx - lag_limit_steps
end_idx = center_idx + lag_limit_steps + 1

limited_lags = lags[start_idx:end_idx]
limited_correlation = correlation[start_idx:end_idx]

plt.figure(figsize=(12, 6))
plt.plot(limited_lags * DT, limited_correlation)
plt.title('Correlación Cruzada entre la Columna 1 y la Columna 2')
plt.xlabel('Desplazamiento Temporal (ms)')
plt.ylabel('Amplitud de Correlación')
plt.grid(True)
plt.axvline(0, color='r', linestyle='--', label='Lag 0')
plt.legend()
plt.savefig('cross_correlation.png')
print("Análisis de correlación cruzada guardado en 'cross_correlation.png'.")

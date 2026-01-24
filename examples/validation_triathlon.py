# examples/validation_triathlon.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.fhn_simulator import FHNNetworkSimulator

# --- Parámetros de Simulación ---
N_COLUMNS = 4
SIMULATION_STEPS = 500
DT = 0.1
COUPLING_STRENGTH = 0.8
SYNAPTIC_DELAY_MS = 2.0
NOISE_LEVEL = 0.1
STIMULUS_CURRENT = 2.0
LAYER_SIZES = {'L4': 5, 'L2/3': 5, 'L5/6': 5}

# --- Parámetros de la Prueba de Información ---
INPUT_PATTERN = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
PATTERN_STEP_DURATION = 20  # ms

# --- Conversión de Parámetros ---
DELAY_STEPS = int(SYNAPTIC_DELAY_MS / DT)

# --- 1. Construcción de la Red en Cadena ---
columns = [CompartmentalColumn(index=i, n_nodes_per_layer=LAYER_SIZES) for i in range(N_COLUMNS)]
simulator = FHNNetworkSimulator(columns, coupling_strength=COUPLING_STRENGTH, delay_steps=DELAY_STEPS)

# --- Contenedores para el Análisis ---
spike_history = np.zeros((SIMULATION_STEPS, N_COLUMNS))
voltage_history_L56 = np.zeros((SIMULATION_STEPS, N_COLUMNS))
voltage_history_col1 = {name: np.zeros((SIMULATION_STEPS, size)) for name, size in LAYER_SIZES.items()}


# --- 2. Implementación de las Pruebas ---

def get_stimulus(step):
    """Genera el estímulo externo y el ruido para cada paso de tiempo."""
    I_ext = np.zeros(LAYER_SIZES['L4'])
    pattern_index = step // PATTERN_STEP_DURATION
    if pattern_index < len(INPUT_PATTERN) and INPUT_PATTERN[pattern_index] == 1:
        I_ext += STIMULUS_CURRENT

    I_noise = {i: np.random.normal(0, NOISE_LEVEL, sum(LAYER_SIZES.values())) for i in range(N_COLUMNS)}

    return {0: I_ext}, I_noise

# --- 3. Bucle Principal de Simulación ---
print("Ejecutando el Triatlón de Validación...")
for i in range(SIMULATION_STEPS):
    external_stimuli, noise_stimuli = get_stimulus(i)
    simulator.run_step(external_stimuli=external_stimuli, noise_stimuli=noise_stimuli)

    # Registrar datos para el análisis
    for j, col in enumerate(columns):
        spike_history[i, j] = 1 if np.any(col.layers['L5/6'].states == 1) else 0
        voltage_history_L56[i, j] = np.mean(col.layers['L5/6'].v)
        if j == 0:
            for name, layer in col.layers.items():
                voltage_history_col1[name][i, :] = layer.v

print("Simulación completada.")

# --- 4. Análisis y Generación del Reporte ---

# Prueba de Flujo
spike_times = [np.where(spike_history[:, i] == 1)[0] for i in range(N_COLUMNS)]
first_spike_times = [t[0] * DT if len(t) > 0 else -1 for t in spike_times]
delays = np.diff(first_spike_times)
avg_delay = np.mean(delays[delays > 0]) if len(delays[delays > 0]) > 0 else "N/A"

# Prueba de Ruido
signal_power = np.mean(np.square(voltage_history_L56))
noise_power = np.var(voltage_history_L56)
snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else "Inf"

# Prueba de Información
input_spikes_flat = np.zeros(SIMULATION_STEPS)
pattern_duration = len(INPUT_PATTERN) * PATTERN_STEP_DURATION
input_spikes_flat[:pattern_duration] = np.array([p for p in INPUT_PATTERN for _ in range(PATTERN_STEP_DURATION)])
output_spikes_c4 = spike_history[:, -1]
hamming_distance = np.sum(input_spikes_flat != output_spikes_c4)
hamming_error_rate = (hamming_distance / len(input_spikes_flat)) * 100

# Visualización Final (2 paneles)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
time_axis = np.arange(SIMULATION_STEPS) * DT

# Panel 1: Trazas de Voltaje de la Columna 1
for name, history in voltage_history_col1.items():
    ax1.plot(time_axis, np.mean(history, axis=1), label=f'C1 - {name}')
ax1.set_title('Dinámica Interna de la Columna 1')
ax1.set_ylabel('Potencial Promedio (v)')
ax1.legend()
ax1.grid(True)

# Panel 2: Raster Plot de Spikes de Todas las Columnas
spike_events = np.where(spike_history == 1)
ax2.scatter(spike_events[0] * DT, spike_events[1] + 1, marker='|', color='black')
ax2.set_title('Propagación de Spikes a Través de la Red')
ax2.set_ylabel('Columna #')
ax2.set_xlabel('Tiempo (ms)')
ax2.set_yticks(range(1, N_COLUMNS + 1))
ax2.grid(True)

plt.tight_layout()
plt.savefig('triathlon_validation_results.png')
print("\nGráfica de validación guardada en 'triathlon_validation_results.png'")

print("\n--- Tablero de Control del Triatlón ---")
print(f"| Prueba      | Métrica                 | Resultado                                            |")
print(f"|-------------|-------------------------|------------------------------------------------------|")
print(f"| Flujo       | Delay Prom./Columna (ms)| {avg_delay:.2f}                                              |")
print(f"| Ruido       | SNR (dB)                | {snr:.2f}                                                |")
print(f"| Información | Error de Hamming (%)    | {hamming_error_rate:.2f}                                         |")

# examples/validation_triathlon.py

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.fhn_simulator import FHNNetworkSimulator

# --- Parámetros de Simulación ---
N_COLUMNS = 4
SIMULATION_STEPS = 5000 # 500 ms
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

# --- 1. Construcción de la Red en Anillo ---
columns = [CompartmentalColumn(index=i, n_nodes_per_layer=LAYER_SIZES) for i in range(N_COLUMNS)]
edges = [(i, (i + 1) % N_COLUMNS) for i in range(N_COLUMNS)] # C1->C2, C2->C3, C3->C4, C4->C1
simulator = FHNNetworkSimulator(columns, edges, coupling_strength=COUPLING_STRENGTH, delay_steps=DELAY_STEPS)

# --- Contenedores para el Análisis ---
spike_history = np.zeros((SIMULATION_STEPS, N_COLUMNS))
voltage_history_L56 = np.zeros((SIMULATION_STEPS, N_COLUMNS))
voltage_history_col1 = {name: np.zeros((SIMULATION_STEPS, size)) for name, size in LAYER_SIZES.items()}


# --- 2. Funciones de Análisis ---

def project_to_discrete_states(spike_hist, window_size_ms, dt_ms):
    """Proyecta un historial de spikes a estados discretos en ventanas de tiempo."""
    window_size_steps = int(window_size_ms / dt_ms)
    n_steps, n_vars = spike_hist.shape
    n_windows = n_steps // window_size_steps

    discrete_states = np.zeros((n_windows, n_vars), dtype=int)
    for i in range(n_windows):
        start = i * window_size_steps
        end = start + window_size_steps
        window_activity = np.sum(spike_hist[start:end, :], axis=0)
        discrete_states[i, :] = (window_activity > 0).astype(int)

    return discrete_states

def find_boolean_attractor(state_sequence):
    """Encuentra el primer ciclo límite en una secuencia de estados booleanos."""
    history = {}
    for i, state in enumerate(state_sequence):
        state_tuple = tuple(state)
        if state_tuple in history:
            start_index = history[state_tuple]
            cycle = state_sequence[start_index:i]
            # Asegurarse de que es un ciclo estable
            if len(cycle) > 0 and np.array_equal(state_sequence[i:i+len(cycle)], cycle):
                return cycle
        history[state_tuple] = i
    return None # No se encontró ciclo

def generate_transition_matrix(state_sequence):
    """Genera una matriz de transición empírica (tabla de verdad) a partir de una secuencia de estados."""
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(state_sequence) - 1):
        from_state = tuple(state_sequence[i])
        to_state = tuple(state_sequence[i+1])
        transitions[from_state][to_state] += 1

    # Formatear para una salida legible
    matrix = {}
    for from_state, to_states in transitions.items():
        # Tomar la transición más frecuente como la canónica
        most_likely_to_state = max(to_states, key=to_states.get)
        matrix[from_state] = most_likely_to_state
    return matrix

# --- 3. Implementación de las Prubas ---

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

# --- 4. Análisis de Correspondencia de Atractores ---

# 1. Proyección a estados discretos
PROJECTION_WINDOW_MS = 5.0
discrete_states = project_to_discrete_states(spike_history, PROJECTION_WINDOW_MS, DT)

# 2. Búsqueda del atractor booleano
boolean_attractor = find_boolean_attractor(discrete_states)

# 3. Generación de la matriz de transición
transition_matrix = generate_transition_matrix(discrete_states)

# 4. Cálculo de A_corr (simplificado)
# Identificar el ciclo límite en los voltajes continuos (buscando picos)
from scipy.signal import find_peaks
peaks, _ = find_peaks(voltage_history_L56[:, 0], height=0)
if len(peaks) > 1:
    continuous_period_ms = np.mean(np.diff(peaks)) * DT
else:
    continuous_period_ms = "N/A"

boolean_period_ms = len(boolean_attractor) * PROJECTION_WINDOW_MS if boolean_attractor is not None else "N/A"

# A_corr: 1 si los períodos coinciden, 0 si no (simplificado)
A_corr = "N/A"
if isinstance(continuous_period_ms, float) and isinstance(boolean_period_ms, float):
    # Considerar una correspondencia si los períodos son aproximadamente iguales
    A_corr = 1 if np.isclose(continuous_period_ms, boolean_period_ms, rtol=0.1) else 0

# --- 5. Generación de Resultados ---

print("\n--- Resultados del Análisis de Atractores ---")
print(f"\n[+] Atractor Booleano Identificado:")
if boolean_attractor is not None:
    for state in boolean_attractor:
        print(f"    {list(state)}")
else:
    print("    No se encontró un ciclo límite en la secuencia discreta.")

print(f"\n[+] Correspondencia de Períodos (A_corr):")
print(f"    - Período del Ciclo Límite Continuo: {continuous_period_ms:.2f} ms" if isinstance(continuous_period_ms, float) else f"    - Período del Ciclo Límite Continuo: {continuous_period_ms}")
print(f"    - Período del Atractor Booleano:     {boolean_period_ms:.2f} ms" if isinstance(boolean_period_ms, float) else f"    - Período del Atractor Booleano:     {boolean_period_ms}")
print(f"    - Índice de Correspondencia (A_corr): {A_corr}")

print("\n[+] Matriz de Transición Empírica (Extracto):")
for i, (from_state, to_state) in enumerate(transition_matrix.items()):
    if i >= 10: # Mostrar solo las primeras 10 transiciones para brevedad
        print("    ...")
        break
    print(f"    {list(from_state)} -> {list(to_state)}")

# --- 6. Generación de Gráfica ---
print("\nGenerando gráfica de resultados...")
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
time_axis_ms = np.arange(SIMULATION_STEPS) * DT

# Subplot 1: Voltaje de la capa de salida (L5/6)
for i in range(N_COLUMNS):
    axes[0].plot(time_axis_ms, voltage_history_L56[:, i], label=f'Columna {i+1} L5/6')
axes[0].set_title('Dinámica Continua: Potencial de Membrana Promedio (Capa L5/6)')
axes[0].set_ylabel('Voltaje (v)')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Estados discretos proyectados
discrete_time_axis_ms = np.arange(len(discrete_states)) * PROJECTION_WINDOW_MS
axes[1].imshow(discrete_states.T, aspect='auto', interpolation='none',
               extent=[0, time_axis_ms[-1], -0.5, N_COLUMNS - 0.5],
               cmap='Greys')
axes[1].set_title(f'Dinámica Discreta: Estados Proyectados (Ventana de {PROJECTION_WINDOW_MS} ms)')
axes[1].set_xlabel('Tiempo (ms)')
axes[1].set_ylabel('Columna')
axes[1].set_yticks(np.arange(N_COLUMNS))
axes[1].set_yticklabels([f'Col {i+1}' for i in range(N_COLUMNS)])

plt.tight_layout()
plt.savefig('attractor_analysis.png')
print("Gráfica guardada en 'attractor_analysis.png'")

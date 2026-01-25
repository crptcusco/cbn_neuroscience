# examples/stochastic_cv_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

# --- 1. Funciones de Análisis ---

def calculate_cv(spike_train, dt):
    """
    Calcula el Coeficiente de Variación (CV) de un tren de spikes.
    CV = std(ISI) / mean(ISI) (Eq. 5.32, Trappenberg)
    """
    spike_times = np.where(spike_train)[0] * dt
    if len(spike_times) < 2:
        return 0.0  # No se pueden calcular intervalos con menos de 2 spikes

    isi = np.diff(spike_times)

    if np.mean(isi) == 0:
        return 0.0

    return np.std(isi) / np.mean(isi)

# --- 2. Parámetros de Simulación y Calibración ---
DT = 0.1  # ms
SIMULATION_TIME_MS = 2000  # Simulación más larga para obtener estadísticas fiables
N_STEPS = int(SIMULATION_TIME_MS / DT)

LAYER_SIZES = {'L4': 10, 'L2/3': 10, 'L5/6': 10} # Más neuronas para mejor estadística
SRM_PARAMS = {'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'dt': DT}

# Parámetros del estímulo y objetivo de CV
STIMULUS_STRENGTH = 18.0  # Estímulo constante para inducir una tasa de disparo estable
CV_TARGET_MIN = 0.8
CV_TARGET_MAX = 1.1

# --- 3. Bucle de Calibración de Ruido ---
print("Iniciando calibración de ruido para alcanzar un CV en [0.8, 1.1]...")
sigma_noise = 0.5  # Valor inicial de la desviación estándar del ruido
cv = 0.0

for i in range(20): # Limitar a 20 iteraciones para evitar bucles infinitos
    # --- Configuración del modelo ---
    column = CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, srm_params=SRM_PARAMS)

    # --- Simulación ---
    l4_spike_history = np.zeros((N_STEPS, LAYER_SIZES['L4']), dtype=bool)

    for step in range(N_STEPS):
        ext_input = np.full(LAYER_SIZES['L4'], STIMULUS_STRENGTH)
        noise_input = np.random.normal(0, sigma_noise, sum(LAYER_SIZES.values()))
        inter_col_input = np.zeros(LAYER_SIZES['L2/3'])

        column.update(ext_input, noise_input, inter_col_input)
        l4_spike_history[step, :] = column.layers['L4'].spikes

    # --- Cálculo del CV ---
    # Calcular el CV promedio sobre todas las neuronas de la capa L4
    cv_per_neuron = [calculate_cv(l4_spike_history[:, i], DT) for i in range(LAYER_SIZES['L4'])]
    cv = np.mean([c for c in cv_per_neuron if c > 0]) # Evitar divisiones por cero

    print(f"Iteración {i+1}: sigma_noise = {sigma_noise:.2f}, CV = {cv:.2f}")

    # --- Ajuste de sigma_noise ---
    if CV_TARGET_MIN <= cv <= CV_TARGET_MAX:
        print("\nCalibración exitosa!")
        break
    elif cv < CV_TARGET_MIN:
        sigma_noise += 0.1  # Aumentar ruido para aumentar CV
    else:
        sigma_noise -= 0.05 # Reducir ruido para disminuir CV (ajuste más fino)

# --- 4. Resultados Finales de la Calibración ---
print("\n--- Resultados de la Calibración ---")
print(f"Sigma de ruido óptimo: {sigma_noise:.2f}")
print(f"CV final de la capa L4: {cv:.2f}")
if not (CV_TARGET_MIN <= cv <= CV_TARGET_MAX):
    print("\nAdvertencia: El CV final está fuera del rango objetivo.")
    print("Puede ser necesario ajustar los parámetros de simulación o calibración.")

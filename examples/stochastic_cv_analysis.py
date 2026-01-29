# examples/stochastic_cv_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.srm_nodegroup import SRM_NodeGroup

# --- 1. Funciones de Análisis ---

def calculate_cv(spike_train, dt):
    """
    Calcula el Coeficiente de Variación (CV) de un tren de spikes.
    CV = std(ISI) / mean(ISI) (Eq. 5.32, Trappenberg)
    """
    spike_times = np.where(spike_train)[0] * dt
    if len(spike_times) < 2:
        return 0.0

    isi = np.diff(spike_times)

    if np.mean(isi) == 0:
        return 0.0

    return np.std(isi) / np.mean(isi)

# --- 2. Parámetros de Simulación y Calibración ---
DT = 0.1
SIMULATION_TIME_MS = 2000
N_STEPS = int(SIMULATION_TIME_MS / DT)
N_NEURONS = 10

SRM_PARAMS = {'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'dt': DT}
STIMULUS_STRENGTH = 18.0
CV_TARGET_MIN = 0.8
CV_TARGET_MAX = 1.1

# --- 3. Bucle de Calibración de Ruido ---
print("Iniciando calibración de ruido para alcanzar un CV en [0.8, 1.1]...")
sigma_noise = 0.5
cv = 0.0

for i in range(20):
    # --- Configuración del modelo ---
    srm_group = SRM_NodeGroup(n_nodes=N_NEURONS, **SRM_PARAMS)

    # --- Simulación ---
    spike_history = np.zeros((N_STEPS, N_NEURONS), dtype=bool)

    for step in range(N_STEPS):
        ext_input = np.full(N_NEURONS, STIMULUS_STRENGTH)
        noise_input = np.random.normal(0, sigma_noise, N_NEURONS)

        srm_group.update(weighted_input_spikes=ext_input, noise_term=noise_input)
        spike_history[step, :] = srm_group.spikes

    # --- Cálculo del CV ---
    cv_per_neuron = [calculate_cv(spike_history[:, i], DT) for i in range(N_NEURONS)]
    cv = np.mean([c for c in cv_per_neuron if c > 0])

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

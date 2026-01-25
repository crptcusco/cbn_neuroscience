# examples/step_response_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

# --- 1. Funciones de Análisis ---

def calculate_population_activity(spike_history, dt_ms, sigma_ms=2.0):
    """
    Calcula la actividad de población A(t) suavizada con un kernel gaussiano.
    """
    raw_activity = np.mean(spike_history, axis=1)
    sigma_steps = sigma_ms / dt_ms
    return gaussian_filter1d(raw_activity, sigma=sigma_steps)

# --- 2. Parámetros de Simulación ---
DT = 0.1
SIMULATION_TIME_MS = 200
N_STEPS = int(SIMULATION_TIME_MS / DT)

LIF_PARAMS = {
    'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'R_m': 10.0,
    'tau_syn_exc': 5.0, 'E_exc': 0.0, 'delta': 2.0, 'dt': DT
}
LAYER_SIZES = {'L4': 100, 'L2/3': 100, 'L5/6': 100} # Población grande

# --- Configuración del Estímulo de Escalón ---
STEP_TIME_MS = 100
STEP_TIME_STEP = int(STEP_TIME_MS / DT)
STIMULUS_LOW = 0.6  # Corresponde a RI = 11mV con R=10 y I_th=1.5
STIMULUS_HIGH = 1.1 # Corresponde a RI = 16mV con R=10 y I_th=1.5

# El estímulo se trata como un peso de spike
# Para que I = 2, el peso debe ser g = I/(R*(V-E)) ~ 2/(10*(-60-0)) = -0.003
# Es más simple tratarlo como una corriente externa de nuevo.
# Vamos a modificar temporalmente el update de la columna para esta prueba.
# O, mejor, inyectamos una corriente constante a través del ruido. No, es confuso.

# La forma más limpia es mantener el modelo GLIF.
# El "peso" del estímulo incrementará la conductancia.
# Necesitamos calibrar qué pesos corresponden a las corrientes deseadas.
# Por ahora, usaremos valores representativos.
SPK_WEIGHT_LOW = 0.5
SPK_WEIGHT_HIGH = 1.0


# --- 3. Bucle de Calibración de Ruido y Análisis de Tiempo de Subida ---
print("Iniciando análisis de respuesta al escalón...")
optimal_noise_sigma = None

for noise_sigma in np.arange(0.5, 5.0, 0.5):
    print(f"\nProbando con NOISE_SIGMA = {noise_sigma:.2f}...")

    # --- Simulación ---
    column = CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, g_axial=4.0, lif_params=LIF_PARAMS)
    l56_spike_history = np.zeros((N_STEPS, LAYER_SIZES['L5/6']), dtype=bool)

    for step in range(N_STEPS):
        stim_weight = SPK_WEIGHT_LOW if step < STEP_TIME_STEP else SPK_WEIGHT_HIGH
        ext_spikes = np.full(LAYER_SIZES['L4'], stim_weight)

        noise = np.random.normal(0, noise_sigma, sum(LAYER_SIZES.values()))
        inter_spikes = np.zeros(LAYER_SIZES['L2/3'])

        column.update(ext_spikes, noise, inter_spikes)
        l56_spike_history[step, :] = column.layers['L5/6'].spikes

    # --- Cálculo de A(t) y Tiempo de Subida ---
    time_axis = np.arange(N_STEPS) * DT
    population_activity = calculate_population_activity(l56_spike_history, DT, sigma_ms=2.0)

    # Encontrar el valor máximo post-escalón
    post_step_activity = population_activity[STEP_TIME_STEP:]
    max_A = np.max(post_step_activity)

    # Encontrar los puntos del 10% y 90%
    try:
        t_10_idx = np.where(post_step_activity >= 0.1 * max_A)[0][0] + STEP_TIME_STEP
        t_90_idx = np.where(post_step_activity >= 0.9 * max_A)[0][0] + STEP_TIME_STEP

        rise_time_ms = (t_90_idx - t_10_idx) * DT
        print(f"  Tiempo de subida (10%-90%): {rise_time_ms:.2f} ms")

        if rise_time_ms < 3.0:
            optimal_noise_sigma = noise_sigma
            print(f"  -> ¡Respuesta rápida! Nivel de ruido óptimo encontrado.")
            break

    except IndexError:
        print("  -> No se alcanzó suficiente actividad para medir el tiempo de subida.")

if optimal_noise_sigma is None:
    print("\nNo se encontró un nivel de ruido que produjera una respuesta suficientemente rápida.")
else:
    print(f"\nAnálisis completado. Ruido óptimo: {optimal_noise_sigma:.2f}")


# --- 4. Generación de la Gráfica Comparativa (Tarea 3) ---

def rate_model_step_response(t, t_step, I_low, I_high, tau):
    """Respuesta teórica de Eq. 5.49 a un escalón."""
    response = np.zeros_like(t)
    # Asumimos g(x) = x para simplificar la forma de la respuesta
    response[t < t_step] = I_low + (response[0] - I_low) * np.exp(-t[t < t_step]/tau)

    A_at_step = I_low + (response[0] - I_low) * np.exp(-t_step/tau)
    t_after_step = t[t >= t_step] - t_step
    response[t >= t_step] = I_high + (A_at_step - I_high) * np.exp(-t_after_step/tau)
    return response

# Re-ejecutar la simulación con el ruido óptimo para obtener los datos para la gráfica
if optimal_noise_sigma is not None:
    print("\nGenerando gráfica comparativa...")
    column = CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, g_axial=4.0, lif_params=LIF_PARAMS)
    l56_spike_history_final = np.zeros((N_STEPS, LAYER_SIZES['L5/6']), dtype=bool)
    for step in range(N_STEPS):
        stim_weight = SPK_WEIGHT_LOW if step < STEP_TIME_STEP else SPK_WEIGHT_HIGH
        ext_spikes = np.full(LAYER_SIZES['L4'], stim_weight)
        noise = np.random.normal(0, optimal_noise_sigma, sum(LAYER_SIZES.values()))
        inter_spikes = np.zeros(LAYER_SIZES['L2/3'])
        column.update(ext_spikes, noise, inter_spikes)
        l56_spike_history_final[step, :] = column.layers['L5/6'].spikes

    # Calcular la actividad empírica
    time_axis = np.arange(N_STEPS) * DT
    empirical_activity = calculate_population_activity(l56_spike_history_final, DT, sigma_ms=2.0)

    # Calcular la respuesta teórica del modelo de tasa
    # Normalizamos para que las alturas coincidan y poder comparar la velocidad
    theoretical_response = rate_model_step_response(time_axis, STEP_TIME_MS,
                                                     np.mean(empirical_activity[:STEP_TIME_STEP]),
                                                     np.max(empirical_activity),
                                                     tau=10.0)

    # Graficar
    plt.figure(figsize=(12, 7))
    plt.plot(time_axis, empirical_activity, label=f'Red de Spikes (Ruido σ={optimal_noise_sigma:.1f})', linewidth=2)
    plt.plot(time_axis, theoretical_response, 'r--', label='Modelo de Tasa Teórico (τ=10ms)')
    plt.axvline(STEP_TIME_MS, color='k', linestyle=':', label='Salto de Estímulo')
    plt.title('Respuesta al Escalón: Red de Spikes vs. Modelo de Tasa')
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Actividad de Población Normalizada A(t)')
    plt.legend()
    plt.grid(True)
    plt.savefig('step_response_comparison.png')
    print("Gráfica guardada en 'step_response_comparison.png'")

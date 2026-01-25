# examples/rate_model_comparison.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

# --- 1. Funciones de Análisis y Modelo Teórico ---

def calculate_population_activity(spike_history, dt_ms, sigma_ms=2.0):
    """
    Calcula la actividad de población A(t) suavizada con un kernel gaussiano.
    """
    raw_activity = np.mean(spike_history, axis=1)
    sigma_steps = sigma_ms / dt_ms
    return gaussian_filter1d(raw_activity, sigma=sigma_steps)

def leaky_integrator_model(t, I_eff, tau_A):
    """
    Solución a la Eq. 5.49 de Trappenberg para un pulso de corriente.
    Describe la evolución de la actividad de población A(t).
    """
    # Esta es una simplificación: asumimos que la actividad decae
    # exponencialmente tras el fin del estímulo.
    # El ajuste encontrará la mejor tau_A para describir este decaimiento.
    # La subida se modelará por el propio pulso.
    # A(t) = A_0 * exp(-t / tau_A)
    # Para el ajuste, necesitamos una forma más general.

    # La ecuación es: tau_A * dA/dt = -A + g(I_ext)
    # Para un pulso I_ext, A(t) subirá y luego decaerá.
    # Vamos a ajustar la fase de decaimiento.

    # Modelo simplificado para ajuste con curve_fit:
    # A(t) = I_eff * (1 - exp(-t/tau_A))  (fase de subida)
    # curve_fit no maneja bien pulsos, así que modelaremos la respuesta general.

    # Usaremos una solución analítica para un pulso rectangular
    # que empieza en t=0 y termina en t=pulse_duration.
    # Esto es complejo para curve_fit. En su lugar, ajustaremos
    # la fase de subida y decaimiento por separado.

    # Para simplificar, ajustaremos un modelo de subida y decaimiento simple.
    # Esta función se usará en el paso 2
    pass

def leaky_integrator_decay(t, A0, tau_A):
    """Modelo exponencial simple para la fase de decaimiento."""
    return A0 * np.exp(-t / tau_A)

# --- 2. Simulación y Ajuste del Modelo ---

# --- Parámetros de Simulación ---
DT = 0.1
SIMULATION_TIME_MS = 500
N_STEPS = int(SIMULATION_TIME_MS / DT)

LIF_PARAMS = {
    'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'R_m': 10.0,
    'tau_syn_exc': 5.0, 'E_exc': 0.0, 'delta': 2.0, 'dt': DT
}
LAYER_SIZES = {'L4': 50, 'L2/3': 50, 'L5/6': 50}

# --- Configuración del Pulso de Estímulo ---
STIMULUS_WEIGHT = 0.8
PULSE_START_MS = 50
PULSE_END_MS = 200
PULSE_START_STEP = int(PULSE_START_MS / DT)
PULSE_END_STEP = int(PULSE_END_MS / DT)

# --- Bucle de Simulación ---
print("Ejecutando simulación de una columna para Tarea 1...")
column = CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, g_axial=4.0, lif_params=LIF_PARAMS)
l56_spike_history = np.zeros((N_STEPS, LAYER_SIZES['L5/6']), dtype=bool)

for step in range(N_STEPS):
    ext_spikes = np.zeros(LAYER_SIZES['L4'])
    if PULSE_START_STEP <= step < PULSE_END_STEP:
        ext_spikes.fill(STIMULUS_WEIGHT)

    noise = np.random.normal(0, 1.0, sum(LAYER_SIZES.values()))

    column.update(ext_spikes, noise, np.zeros(LAYER_SIZES['L2/3']))
    l56_spike_history[step, :] = column.layers['L5/6'].spikes

# --- 3. Cálculo de A(t) y Ajuste del Modelo ---
print("Calculando actividad de población y ajustando el modelo de tasa...")
time_axis = np.arange(N_STEPS) * DT
empirical_A = calculate_population_activity(l56_spike_history, DT, sigma_ms=5.0)

# Aislar la fase de decaimiento para el ajuste
decay_start_step = PULSE_END_STEP + int(10 / DT) # Empezar ajuste un poco después del pulso
decay_time = time_axis[decay_start_step:] - time_axis[decay_start_step]
decay_data = empirical_A[decay_start_step:]

# Ajustar el modelo a los datos de decaimiento
initial_guess = [np.max(decay_data), 20.0] # A0_guess, tau_A_guess (ms)
params, _ = curve_fit(leaky_integrator_decay, decay_time, decay_data, p0=initial_guess)
A0_fit, tau_A_fit = params

# Generar la curva teórica ajustada
fitted_decay = leaky_integrator_decay(decay_time, A0_fit, tau_A_fit)

# Calcular el Error Cuadrático Medio (MSE)
mse = np.mean((decay_data - fitted_decay)**2)

print("\n--- Resultados del Ajuste del Modelo ---")
print(f"Parámetros ajustados: A0 = {A0_fit:.4f}, tau_A = {tau_A_fit:.2f} ms")
print(f"Error Cuadrático Medio (MSE) en la fase de decaimiento: {mse:.6f}")


# --- 4. Test de Estado Estacionario (Tarea 2) ---

print("\nEjecutando simulación de estado estacionario para Tarea 2...")
SIM_TIME_STEADY_MS = 1000
N_STEPS_STEADY = int(SIM_TIME_STEADY_MS / DT)
STIMULUS_WEIGHT_STEADY = 0.6 # Un estímulo constante

column_steady = CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, g_axial=4.0, lif_params=LIF_PARAMS)
l56_spikes_steady = np.zeros((N_STEPS_STEADY, LAYER_SIZES['L5/6']), dtype=bool)

for step in range(N_STEPS_STEADY):
    ext_spikes = np.full(LAYER_SIZES['L4'], STIMULUS_WEIGHT_STEADY)
    noise = np.random.normal(0, 1.5, sum(LAYER_SIZES.values()))
    column_steady.update(ext_spikes, noise, np.zeros(LAYER_SIZES['L2/3']))
    l56_spikes_steady[step, :] = column_steady.layers['L5/6'].spikes

# Calcular y analizar la actividad de población en estado estacionario
time_axis_steady = np.arange(N_STEPS_STEADY) * DT
steady_A = calculate_population_activity(l56_spikes_steady, DT, sigma_ms=5.0)

# Analizar la última mitad de la simulación para ver si es estable u oscilatoria
analysis_window = steady_A[N_STEPS_STEADY//2:]
mean_A = np.mean(analysis_window)
std_A = np.std(analysis_window)

print("\n--- Resultados del Test de Estado Estacionario ---")
print(f"Actividad de población media (últimos 500ms): {mean_A:.4f}")
print(f"Desviación estándar de la actividad: {std_A:.4f}")

if std_A / mean_A > 0.1: # Si la desviación es >10% de la media, consideramos que oscila
    print("Resultado: La actividad de la población es OSCILATORIA.")
    # (El cálculo de la frecuencia se podría añadir aquí si es necesario)
else:
    print("Resultado: La actividad de la población alcanza un ESTADO ESTACIONARIO estable.")


# --- 5. Caracterización de la Función de Activación g(x) (Tarea 3) ---

print("\nCaracterizando la función de activación g(x) para Tarea 3...")
input_strengths = np.linspace(0.1, 1.5, 15)
output_activities = []

for strength in input_strengths:
    column_g = CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, g_axial=4.0, lif_params=LIF_PARAMS)
    spikes_g = np.zeros((N_STEPS_STEADY, LAYER_SIZES['L5/6']), dtype=bool)

    for step in range(N_STEPS_STEADY):
        ext_spikes = np.full(LAYER_SIZES['L4'], strength)
        noise = np.random.normal(0, 1.5, sum(LAYER_SIZES.values()))
        column_g.update(ext_spikes, noise, np.zeros(LAYER_SIZES['L2/3']))
        spikes_g[step, :] = column_g.layers['L5/6'].spikes

    activity_g = calculate_population_activity(spikes_g, DT, sigma_ms=5.0)
    mean_activity = np.mean(activity_g[N_STEPS_STEADY//2:])
    output_activities.append(mean_activity)
    print(f"  I_ext = {strength:.2f} -> A = {mean_activity:.4f}")

# --- Generar Gráfica de g(x) ---
plt.figure(figsize=(10, 6))
plt.plot(input_strengths, output_activities, 'o-', label='g(x) empírica')
plt.title('Función de Activación de la Población g(x)')
plt.xlabel('Intensidad del Estímulo de Entrada (I_ext)')
plt.ylabel('Actividad de Población Estacionaria (A)')
plt.grid(True)
plt.legend()
plt.savefig('activation_function_g_x.png')
print("\nGráfica de la función de activación guardada en 'activation_function_g_x.png'")


# --- 6. Generar Gráfica Principal A(t) (Entrega) ---
plt.figure(figsize=(12, 7))
plt.plot(time_axis, empirical_A, label='Actividad de Población Empírica A(t)')
plt.plot(time_axis[decay_start_step:], fitted_decay, 'r--', label=f'Ajuste Teórico (τ_eff = {tau_A_fit:.2f} ms)')
plt.axvspan(PULSE_START_MS, PULSE_END_MS, color='gray', alpha=0.2, label='Pulso de Estímulo')
plt.title('Dinámica de la Actividad de Población A(t) y Ajuste del Modelo de Tasa')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Actividad de Población A(t)')
plt.legend()
plt.grid(True)
plt.savefig('population_dynamics_A_t.png')
print("\nGráfica principal de la dinámica de población guardada en 'population_dynamics_A_t.png'")

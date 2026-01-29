# examples/spike_vs_rate_model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

# --- 1. Funciones de Análisis y Modelo Teórico ---

def calculate_population_activity(spike_history, dt_ms, sigma_ms=2.0):
    raw_activity = np.mean(spike_history, axis=1)
    sigma_steps = sigma_ms / dt_ms
    return gaussian_filter1d(raw_activity, sigma=sigma_steps)

def leaky_integrator_decay(t, A0, tau_A):
    return A0 * np.exp(-t / tau_A)

# --- 2. Simulación y Ajuste del Modelo ---
DT = 0.1
SIMULATION_TIME_MS = 500
N_STEPS = int(SIMULATION_TIME_MS / DT)

LIF_PARAMS = {
    'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'R_m': 10.0,
    'tau_syn_exc': 5.0, 'delta': 2.0, 'dt': DT
}
LAYER_SIZES = {'L4': 50, 'L5/6': 50}

PULSE_START_MS = 50
PULSE_END_MS = 200
PULSE_START_STEP = int(PULSE_START_MS / DT)
PULSE_END_STEP = int(PULSE_END_MS / DT)

print("Ejecutando simulación de una columna para Tarea 1...")
rules = [{'sources': [(0, 'L4')], 'target_col': 0, 'target_layer': 'L5/6', 'weight': 1.2}]
columns = [CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, model_class=LIF_NodeGroup, model_params=LIF_PARAMS)]
simulator = NetworkSimulator(columns, rules)
l56_spike_history = np.zeros((N_STEPS, LAYER_SIZES['L5/6']), dtype=bool)

for step in range(N_STEPS):
    ext_inputs = {}
    if PULSE_START_STEP <= step < PULSE_END_STEP:
        ext_inputs = {0: {'L4': {'exc_spikes': 0.8, 'I_noise': np.random.normal(0, 1.0, LAYER_SIZES['L4'])}}}
    else:
        ext_inputs = {0: {'L4': {'I_noise': np.random.normal(0, 1.0, LAYER_SIZES['L4'])}}}

    simulator.run_step(step, ext_inputs)
    l56_spike_history[step, :] = columns[0].layers['L5/6'].spikes

print("Calculando actividad de población y ajustando el modelo de tasa...")
time_axis = np.arange(N_STEPS) * DT
empirical_A = calculate_population_activity(l56_spike_history, DT, sigma_ms=5.0)

decay_start_step = PULSE_END_STEP + int(10 / DT)
decay_time = time_axis[decay_start_step:] - time_axis[decay_start_step]
decay_data = empirical_A[decay_start_step:]

params, _ = curve_fit(leaky_integrator_decay, decay_time, decay_data, p0=[np.max(decay_data), 20.0])
A0_fit, tau_A_fit = params
fitted_decay = leaky_integrator_decay(decay_time, A0_fit, tau_A_fit)
mse = np.mean((decay_data - fitted_decay)**2)

print(f"Resultados del Ajuste: A0 = {A0_fit:.4f}, tau_A = {tau_A_fit:.2f} ms, MSE = {mse:.6f}")

# --- Test de Estado Estacionario ---
print("\nEjecutando simulación de estado estacionario para Tarea 2...")
SIM_TIME_STEADY_MS = 1000
N_STEPS_STEADY = int(SIM_TIME_STEADY_MS / DT)

columns_steady = [CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, model_class=LIF_NodeGroup, model_params=LIF_PARAMS)]
simulator_steady = NetworkSimulator(columns_steady, rules)
l56_spikes_steady = np.zeros((N_STEPS_STEADY, LAYER_SIZES['L5/6']), dtype=bool)

for step in range(N_STEPS_STEADY):
    ext_inputs = {0: {'L4': {'exc_spikes': 0.6, 'I_noise': np.random.normal(0, 1.5, LAYER_SIZES['L4'])}}}
    simulator_steady.run_step(step, ext_inputs)
    l56_spikes_steady[step, :] = columns_steady[0].layers['L5/6'].spikes

steady_A = calculate_population_activity(l56_spikes_steady, DT, sigma_ms=5.0)
analysis_window = steady_A[N_STEPS_STEADY//2:]
mean_A, std_A = np.mean(analysis_window), np.std(analysis_window)
print(f"Resultados Estado Estacionario: Media A = {mean_A:.4f}, Std Dev A = {std_A:.4f}")

# --- Caracterización de g(x) ---
print("\nCaracterizando la función de activación g(x) para Tarea 3...")
input_strengths = np.linspace(0.1, 1.5, 15)
output_activities = []

for strength in input_strengths:
    cols_g = [CompartmentalColumn(index=0, n_nodes_per_layer=LAYER_SIZES, model_class=LIF_NodeGroup, model_params=LIF_PARAMS)]
    sim_g = NetworkSimulator(cols_g, rules)
    spikes_g = np.zeros((N_STEPS_STEADY, LAYER_SIZES['L5/6']), dtype=bool)

    for step in range(N_STEPS_STEADY):
        ext_inputs = {0: {'L4': {'exc_spikes': strength, 'I_noise': np.random.normal(0, 1.5, LAYER_SIZES['L4'])}}}
        sim_g.run_step(step, ext_inputs)
        spikes_g[step, :] = cols_g[0].layers['L5/6'].spikes

    activity_g = calculate_population_activity(spikes_g, DT, sigma_ms=5.0)
    output_activities.append(np.mean(activity_g[N_STEPS_STEADY//2:]))

# --- Visualizaciones ---
plt.figure(figsize=(10, 6))
plt.plot(input_strengths, output_activities, 'o-')
plt.title('Función de Activación Empírica g(x)')
plt.xlabel('Intensidad del Estímulo (I_ext)')
plt.ylabel('Actividad de Población Estacionaria (A)')
plt.grid(True)
plt.savefig('activation_function_g_x.png')

plt.figure(figsize=(12, 7))
plt.plot(time_axis, empirical_A, label='Actividad Empírica A(t)')
plt.plot(time_axis[decay_start_step:], fitted_decay, 'r--', label=f'Ajuste (τ_eff = {tau_A_fit:.2f} ms)')
plt.axvspan(PULSE_START_MS, PULSE_END_MS, color='gray', alpha=0.2, label='Estímulo')
plt.title('Dinámica de Actividad de Población A(t)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Actividad A(t)')
plt.legend()
plt.grid(True)
plt.savefig('population_dynamics_A_t.png')

print("\nGráficas guardadas.")

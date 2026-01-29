# examples/population_dynamics_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.glif_simulator import GLIFNetworkSimulator

# --- 1. Parámetros de Simulación ---
DT = 0.1  # ms
SIMULATION_TIME_MS = 2000  # 2 segundos para un buen análisis espectral
N_STEPS = int(SIMULATION_TIME_MS / DT)
N_COLUMNS = 4

from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.network_simulator import NetworkSimulator

# --- Parámetros del Modelo LIF ---
LIF_PARAMS = {
    'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'R_m': 10.0,
    'tau_syn_exc': 5.0, 'delta': 2.0, 'dt': DT
}
LAYER_SIZES = {'L4': 20, 'L2/3': 20, 'L5/6': 20} # Poblaciones más grandes

# --- Parámetros de Red y Estímulo ---
COUPLING_STRENGTH = 4.0
STIMULUS_SPIKE_WEIGHT = 0.8
NOISE_SIGMA = 1.5

# --- 2. Construcción de la Red ---
print("Construyendo la red de 4 columnas con la nueva API...")
columns = [CompartmentalColumn(index=i, n_nodes_per_layer=LAYER_SIZES,
                               model_class=LIF_NodeGroup, model_params=LIF_PARAMS)
           for i in range(N_COLUMNS)]

# Convertir edges a rules
rules = []
for i in range(N_COLUMNS):
    target_col = (i + 1) % N_COLUMNS
    # Acoplamiento de L5/6 a L4 de la siguiente columna
    rules.append({'sources': [(i, 'L5/6')], 'target_col': target_col, 'target_layer': 'L4',
                  'type': 'additive', 'weight': COUPLING_STRENGTH})

simulator = NetworkSimulator(columns, rules)

# --- 3. Bucle de Simulación ---
print("Ejecutando simulación...")
# Usaremos el historial de la primera columna como representativo
spike_history_c0 = np.zeros((N_STEPS, LAYER_SIZES['L5/6']))

for step in range(N_STEPS):
    ext_inputs = {}
    for i in range(N_COLUMNS):
        ext_inputs[i] = {
            'L4': {'exc_spikes': STIMULUS_SPIKE_WEIGHT, 'I_noise': np.random.normal(0, NOISE_SIGMA, LAYER_SIZES['L4'])}
        }
    simulator.run_step(step, ext_inputs)
    spike_history_c0[step, :] = columns[0].layers['L5/6'].spikes

print("Simulación completada.")

# --- 4. Funciones de Análisis de Población ---

def calculate_population_activity(spike_history, dt_ms, sigma_ms=2.0):
    """
    Calcula la actividad de población A(t) suavizada con un kernel gaussiano.
    Basado en Eq. 5.46 y 5.47 de Trappenberg.
    """
    # A(t) cruda: fracción de neuronas disparando en cada paso de tiempo
    raw_activity = np.mean(spike_history, axis=1)

    # Suavizado con kernel gaussiano
    # La sigma para el filtro se da en unidades de 'dt'
    sigma_steps = sigma_ms / dt_ms
    smoothed_activity = gaussian_filter1d(raw_activity, sigma=sigma_steps)

    return smoothed_activity

def calculate_power_spectrum(signal, dt_ms):
    """
    Calcula el espectro de potencia de una señal usando FFT.
    """
    n = len(signal)

    # Ignorar la componente DC (media) de la señal
    signal_no_dc = signal - np.mean(signal)

    # Calcular la FFT
    fft_result = np.fft.fft(signal_no_dc)

    # Calcular la Densidad Espectral de Potencia (PSD)
    power_spectrum = np.abs(fft_result)**2 / n

    # Calcular las frecuencias correspondientes
    freqs = np.fft.fftfreq(n, d=dt_ms / 1000.0) # Frecuencia en Hz

    # Devolver solo la mitad positiva del espectro
    return freqs[:n//2], power_spectrum[:n//2]


print("Funciones de análisis de población implementadas.")
# (El bucle de simulación y la visualización se añadirán en los siguientes pasos)

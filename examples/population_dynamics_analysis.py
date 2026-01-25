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

# --- Parámetros del Modelo GLIF ---
LIF_PARAMS = {
    'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'R_m': 10.0,
    'tau_syn_exc': 5.0, 'E_exc': 0.0, 'delta': 2.0, 'dt': DT
}
LAYER_SIZES = {'L4': 20, 'L2/3': 20, 'L5/6': 20} # Poblaciones más grandes

# --- Parámetros de Red y Estímulo ---
COUPLING_STRENGTH = 4.0  # Fuerza de acoplamiento entre columnas
DELAY_MS = 2.0
STIMULUS_SPIKE_WEIGHT = 0.8 # Estímulo tónico a la capa de entrada
NOISE_SIGMA = 1.5 # Nivel de ruido para mantener la actividad

# --- 2. Construcción de la Red ---
print("Construyendo la red de 4 columnas...")
columns = [CompartmentalColumn(index=i, n_nodes_per_layer=LAYER_SIZES, g_axial=4.0, lif_params=LIF_PARAMS)
           for i in range(N_COLUMNS)]
edges = [(i, (i + 1) % N_COLUMNS) for i in range(N_COLUMNS)] # Anillo: C1->C2, C2->C3, ...

simulator = GLIFNetworkSimulator(columns, edges, coupling_strength=COUPLING_STRENGTH, delay_ms=DELAY_MS, dt=DT)

# --- 3. Funciones de Análisis de Población ---

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

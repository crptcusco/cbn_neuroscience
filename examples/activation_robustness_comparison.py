# examples/activation_robustness_comparison.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
# Asumiremos que existe un simulador de red genérico
# from cbn_neuroscience.core.network_simulator import NetworkSimulator

# --- Helper para FFT ---
def calculate_power_spectrum(signal, dt_ms):
    n = len(signal)
    signal_no_dc = signal - np.mean(signal)
    fft_result = np.fft.fft(signal_no_dc)
    power_spectrum = np.abs(fft_result)**2 / n
    freqs = np.fft.fftfreq(n, d=dt_ms / 1000.0)
    return freqs[:n//2], power_spectrum[:n//2]

# --- 1. Función de Simulación y Análisis ---

def run_and_analyze_network(gain_function_type, gain_params):
    """
    Configura, ejecuta y analiza una red de 4 columnas con una
    función de activación específica.
    """
    print(f"\n--- Simulación con g(x) = {gain_function_type} ---")

    # --- Parámetros ---
    DT = 0.1
    SIM_TIME_MS = 2000
    N_STEPS = int(SIM_TIME_MS / DT)
    N_COLUMNS = 4

    rate_params = {
        'tau_A': 20.0, 'dt': DT,
        'gain_function_type': gain_function_type,
        **gain_params
    }

    # --- Construcción de la Red ---
    columns = [CompartmentalColumn(index=i, n_nodes_per_layer={'L4':1, 'L2/3':1, 'L5/6':1},
                                   model_class=RateNodeGroup, model_params=rate_params, g_axial=2.0)
               for i in range(N_COLUMNS)]

    # --- Bucle de Simulación (simplificado, sin simulador de red por ahora) ---
    coupling_strength = 6.0
    ext_current = 4.0

    history_A = {i: np.zeros(N_STEPS) for i in range(N_COLUMNS)}

    for step in range(1, N_STEPS):
        for i, col in enumerate(columns):
            # Acoplamiento de la columna anterior en el anillo
            prev_col_idx = (i - 1 + N_COLUMNS) % N_COLUMNS
            inter_col_input = coupling_strength * columns[prev_col_idx].layers['L5/6'].A

            # El input externo solo a la primera columna para romper la simetría
            current_ext_input = ext_current if i == 0 else 0.0

            col.update(ext_input=current_ext_input, inter_column_input=inter_col_input)

            history_A[i][step] = col.layers['L5/6'].A[0]

    # --- Análisis Espectral ---
    signal_A = history_A[0] # Analizar la primera columna
    freqs, power = calculate_power_spectrum(signal_A, DT)

    return freqs, power

# --- 2. Ejecución de las Simulaciones ---

# Configuración 1: Step Function (sigmoide con beta alto)
params_step = {'x0': 3.0}
freqs_step, power_step = run_and_analyze_network('step', params_step)

# Configuración 2: Threshold-Linear Function
params_thresh = {'theta': 2.0}
freqs_thresh, power_thresh = run_and_analyze_network('threshold_linear', params_thresh)

# Configuración 3: Sigmoid Function (suave)
params_sig = {'beta': 1.5, 'x0': 4.0}
freqs_sig, power_sig = run_and_analyze_network('sigmoid', params_sig)


# --- 3. Análisis de Estabilidad ---

def analyze_peak(freqs, power):
    try:
        # Encontrar el pico en la banda Gamma (30-50 Hz)
        gamma_band = (freqs > 30) & (freqs < 50)
        peak_idx = np.argmax(power[gamma_band])
        peak_freq = freqs[gamma_band][peak_idx]
        peak_power = power[gamma_band][peak_idx]

        # Calcular la calidad (Q-factor simplificado): potencia / anchura a media altura
        half_power = peak_power / 2
        width_indices = np.where(power[gamma_band] > half_power)[0]
        width_hz = (width_indices[-1] - width_indices[0]) * (freqs[1] - freqs[0])
        quality = peak_power / width_hz if width_hz > 0 else 0

        return peak_freq, peak_power, quality
    except (ValueError, IndexError):
        return 0, 0, 0

print("\n--- Análisis de Estabilidad Gamma ---")
freq_s, power_s, quality_s = analyze_peak(freqs_step, power_step)
print(f"g_step:    Pico a {freq_s:.2f} Hz, Potencia={power_s:.4f}, Calidad={quality_s:.4f}")

freq_t, power_t, quality_t = analyze_peak(freqs_thresh, power_thresh)
print(f"g_theta:   Pico a {freq_t:.2f} Hz, Potencia={power_t:.4f}, Calidad={quality_t:.4f}")

freq_sig, power_sig, quality_sig = analyze_peak(freqs_sig, power_sig)
print(f"g_sigmoid: Pico a {freq_sig:.2f} Hz, Potencia={power_sig:.4f}, Calidad={quality_sig:.4f}")

# (La generación de la gráfica final se hará en el siguiente paso)

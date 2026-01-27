# examples/gerstner_activation_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup

# --- 1. Implementación de la Función de Gerstner (Eq. 5.55) ---

def gerstner_activation(x, t_ref, tau):
    """
    Calcula la tasa de disparo de una población según Gerstner & Kistler.
    x es la corriente de entrada normalizada.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # La ecuación original asume que x es R*I.
        # tau * x debe ser > 1 para que el logaritmo sea real.
        # x = (v - v_rest) / R
        # Para nosotros, x es la corriente I. Con R=10, tau=15, umbral -55, reposo -70
        # El umbral de corriente es (theta - v_rest)/R_m = (-55 - (-70))/10 = 1.5

        # El argumento del log es (1 - 1/(tau*x)). Esto requiere tau*x > 1.
        # La formulación en el libro es sutil. Una forma más estándar es:
        # Tasa = 1 / (t_ref + (tau_m / (v_th - v_reset)) * log((I - I_th) / (I - I_th)))
        # Vamos a usar la formulación directa de la Eq. 5.55

        # El input 'x' en la eq. es V_in / V_th. Normalicemos nuestra corriente.
        # I_th = (theta - v_rest) / R_m
        I_th = 1.5
        x_norm = x / I_th

        # Evitar el log de negativo o cero
        arg = 1 - (1 / (tau * x_norm))

        rate = np.where(arg > 0, 1.0 / (t_ref - tau * np.log(arg)), 0)
    return rate * 1000  # Convertir a Hz (1/ms -> 1/s)


# --- 2. Función de Simulación de Población ---

def simulate_population(current, lif_params, sim_time_ms=1000):
    """
    Simula un único LIF_NodeGroup y devuelve su tasa de disparo promedio.
    """
    dt = lif_params['dt']
    n_steps = int(sim_time_ms / dt)
    n_neurons = 50 # Simular una población

    # Utilizar la implementación de la librería
    lif_pop = LIF_NodeGroup(n_nodes=n_neurons, **lif_params)

    total_spikes = 0

    # La corriente se puede pasar como I_noise
    inputs = {'I_noise': np.full(n_neurons, current)}

    for step in range(n_steps):
        lif_pop.update(step * dt, **inputs)
        total_spikes += np.sum(lif_pop.spikes)

    # Tasa de disparo promedio en Hz
    avg_rate = total_spikes / (n_neurons * sim_time_ms / 1000.0)
    return avg_rate


# --- 3. Caracterización de la Tasa de Disparo Empírica ---

print("Caracterizando la tasa de disparo empírica (curva F-I)...")
LIF_PARAMS = {
    'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'R_m': 10.0,
    'delta': 2.0, 'dt': 0.1
}

# Rango de corrientes de entrada para simular
# El umbral teórico es 1.5, así que barremos alrededor de ese valor.
input_currents = np.linspace(1.6, 5.0, 20)
empirical_firing_rates = []

for current in input_currents:
    rate = simulate_population(current, LIF_PARAMS)
    empirical_firing_rates.append(rate)
    print(f"  I_in = {current:.2f} -> Tasa = {rate:.2f} Hz")

print("Caracterización completa.")
# (La comparación y visualización se harán en el siguiente paso)

# examples/mean_field_validation.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup

# --- 1. Parámetros ---
RATE_PARAMS = {
    'tau_A': 100.0,
    'dt': 0.1,
    'gain_function_type': 'gerstner',
    't_ref': 1.0,
    'gerstner_tau': 5.0,
    'I_th': 1.5
}

SIM_TIME_MS = 500  # Tiempo suficiente para alcanzar estado estacionario
N_STEPS = int(SIM_TIME_MS / RATE_PARAMS['dt'])

# Rango de corrientes de entrada para testear
input_currents = np.linspace(0, 5.0, 50)
output_rates = []

# --- 2. Caracterización de la Curva F-I ---
print("Caracterizando la curva F-I del modelo de tasa...")
for current in input_currents:
    # Crear una nueva instancia para cada simulación
    rate_pop = RateNodeGroup(n_nodes=1, **RATE_PARAMS)

    # Simular hasta estado estacionario
    for _ in range(N_STEPS):
        rate_pop.update(I_total=current)

    # La actividad final es la tasa de estado estacionario
    output_rates.append(rate_pop.A[0])

print("Caracterización completa.")

# --- 3. Generación de la Gráfica de Validación ---
plt.figure(figsize=(12, 7))

# Curva empírica (de la simulación de nuestro RateNodeGroup)
plt.plot(input_currents, output_rates, 'b-', linewidth=3, label='Simulación del Modelo de Tasa (Eq. 5.49)')

# Curva teórica (directamente de la función de Gerstner)
# Necesitamos la función de ganancia para plotearla
theoretical_pop = RateNodeGroup(n_nodes=1, **RATE_PARAMS)
theoretical_rates = theoretical_pop.get_gain(input_currents)
plt.plot(input_currents, theoretical_rates, 'r--', label='Función Analítica de Gerstner (Eq. 5.55)')

plt.title('Validación del Modelo de Tasa vs. Datos del Hipocampo (Fig. 5.16B)')
plt.xlabel('Corriente de Entrada (I)')
plt.ylabel('Tasa de Disparo de Salida (Hz)')
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig('mean_field_activation_curve.png')

print("\nGráfica de validación guardada en 'mean_field_activation_curve.png'")

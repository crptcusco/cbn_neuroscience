# examples/nonlinear_coupling_demo.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

# --- 1. Definición de la Red y Reglas ---
rate_params = {'tau_A': 10.0, 'gain_function_type': 'sigmoid', 'beta': 2.0, 'x0': 4.0}
layer_names = ['L23', 'L5', 'L4', 'L6'] # Añadir L4 y L6

# --- Reglas para Lógica AND (Columnas 0 y 1) ---
rule_1 = {'sources': [(0, 'L23')], 'target_col': 0, 'target_layer': 'L5', 'type': 'additive', 'weight': 5.0}
rule_2 = {'sources': [(1, 'L23')], 'target_col': 1, 'target_layer': 'L5', 'type': 'additive', 'weight': 5.0}
rule_3 = {'sources': [(0, 'L23'), (1, 'L5')], 'target_col': 0, 'target_layer': 'L5', 'type': 'multiplicative', 'weight': 20.0}

# --- Reglas para Inhibición Divisiva (Columna 2) ---
# L4 recibe un input externo
# L6 se activa por L4 (feedback loop simple)
rule_4 = {'sources': [(2, 'L4')], 'target_col': 2, 'target_layer': 'L6', 'type': 'additive', 'weight': 1.0}
# L6 inhibe de forma divisiva a L4
rule_5 = {'sources': [(2, 'L6')], 'target_col': 2, 'target_layer': 'L4', 'type': 'divisive', 'weight': 5.0}


# --- Creación de las Columnas ---
columns = [
    CompartmentalColumn(index=0, layer_names=layer_names, coupling_rules=[rule_1, rule_3], rate_params=rate_params),
    CompartmentalColumn(index=1, layer_names=layer_names, coupling_rules=[rule_2], rate_params=rate_params),
    CompartmentalColumn(index=2, layer_names=layer_names, coupling_rules=[rule_4, rule_5], rate_params=rate_params)
]

# --- 2. Simulación ---
DT = 0.1; SIM_TIME_MS = 300; N_STEPS = int(SIM_TIME_MS / DT)
history = {f'col_{i}': {name: np.zeros(N_STEPS) for name in layer_names} for i in range(3)}
network_state = {f'col_{i}': {name: 0.0 for name in layer_names} for i in range(3)}

print("Ejecutando simulación de acoplamiento no lineal...")
for step in range(N_STEPS):
    ext_inputs_t = [{}, {}, {}]

    # --- Estímulos para Lógica AND ---
    if 20 / DT < step < 150 / DT: ext_inputs_t[0] = {'L23': 8.0}
    if 150 / DT < step:
        ext_inputs_t[0] = {'L23': 8.0}
        ext_inputs_t[1] = {'L23': 8.0}

    # --- Estímulos para Inhibición Divisiva ---
    # Input constante a L4 de la Col 3
    if step > 20 / DT:
        ext_inputs_t[2] = {'L4': 10.0}

    # --- Actualizar Red ---
    for i, col in enumerate(columns):
        col.update(network_state=network_state, ext_inputs=ext_inputs_t[i])
    for i, col in enumerate(columns):
        current_activities = col.get_layer_activities()
        network_state[f'col_{i}'] = current_activities
        for layer_name, activity in current_activities.items():
            history[f'col_{i}'][layer_name][step] = activity

# --- 3. Visualización ---
time_axis = np.arange(N_STEPS) * DT
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Lógica AND
axes[0].plot(time_axis, history['col_0']['L5'], label='Col 0 - L5 (Output AND)', linewidth=3)
axes[0].plot(time_axis, history['col_0']['L23'], label='Col 0 - L2/3 (Input A)')
axes[0].plot(time_axis, history['col_1']['L5'], label='Col 1 - L5 (Input B)')
axes[0].set_title('Tarea 1: Lógica Multiplicativa (AND)')
axes[0].legend(); axes[0].grid(True, linestyle='--')
axes[0].axvspan(150, 300, color='lightblue', alpha=0.3)

# Inhibición Divisiva
axes[1].plot(time_axis, history['col_2']['L4'], label='Col 2 - L4 (Input Sensorial)', linewidth=3)
axes[1].plot(time_axis, history['col_2']['L6'], label='Col 2 - L6 (Señal Inhibitoria)')
axes[1].set_title('Tarea 2: Inhibición Divisiva (Shunting)')
axes[1].legend(); axes[1].grid(True, linestyle='--')

# Inputs para referencia
axes[2].plot(time_axis, [ext_inputs_t[0].get('L23', 0) for t in range(N_STEPS)], label='Input a Col 0 L2/3')
axes[2].plot(time_axis, [ext_inputs_t[1].get('L23', 0) for t in range(N_STEPS)], label='Input a Col 1 L2/3')
axes[2].plot(time_axis, [ext_inputs_t[2].get('L4', 0) for t in range(N_STEPS)], label='Input a Col 2 L4')
axes[2].set_title('Estímulos Externos')
axes[2].set_xlabel('Tiempo (ms)'); axes[2].legend(); axes[2].grid(True, linestyle='--')


plt.tight_layout()
plt.savefig('nonlinear_coupling_demo.png')
print("Simulación completada. Gráfica guardada en 'nonlinear_coupling_demo.png'")

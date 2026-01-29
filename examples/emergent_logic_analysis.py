# examples/emergent_logic_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator
from cbn_neuroscience.core.plasticity_manager import PlasticityManager

# --- 1. Configuración de la Red ---
DT = 0.1
N_COLUMNS = 4
N_NEURONS = 10
layer_names = ['L5']

lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
              'R_m': 10.0, 'tau_syn_exc': 5.0, 'tau_syn_inh': 5.0, 'delta': 2.0, 'dt': DT}
stdp_params = {'a_plus': 0.05, 'a_minus': -0.05, 'tau_plus': 20.0, 'tau_minus': 20.0,
               'w_max': 1.0, 'w_min': -1.0}

# Conectividad total inicial con pesos pequeños y positivos
rules = []
for i in range(N_COLUMNS):
    for j in range(N_COLUMNS):
        if i == j: continue
        rules.append({'sources': [(i, 'L5')], 'target_col': j, 'target_layer': 'L5',
                      'type': 'additive', 'weight': 0.2})

columns = [CompartmentalColumn(index=i, n_nodes_per_layer={'L5': N_NEURONS},
                               model_class=LIF_NodeGroup, model_params=lif_params)
           for i in range(N_COLUMNS)]

plasticity = PlasticityManager(rule_type='stdp_multiplicative', **stdp_params)
simulator = NetworkSimulator(columns, rules, plasticity)

# --- 2. Fase de Entrenamiento con Estímulos Aleatorios ---
TRAIN_TIME_MS = 5000
N_STEPS_TRAIN = int(TRAIN_TIME_MS / DT)

print(f"Iniciando entrenamiento de {TRAIN_TIME_MS} ms con estímulos aleatorios...")

# Probabilidad de que una columna reciba un estímulo en un paso de tiempo
stim_probability = 0.05
stim_strength = 2.5

initial_weights = simulator.connection_manager.weights.copy()

for step in range(N_STEPS_TRAIN):
    ext_inputs = {}
    for i in range(N_COLUMNS):
        if np.random.rand() < stim_probability:
            ext_inputs[i] = {'L5': {'exc_spikes': stim_strength}}

    # Añadir ruido de fondo
    noise_inputs = {i: {'L5': {'I_noise': np.random.normal(0, 1.0, N_NEURONS)}} for i in range(N_COLUMNS)}

    # Combinar inputs
    for i in range(N_COLUMNS):
        if i in ext_inputs:
             ext_inputs[i]['L5']['I_noise'] = noise_inputs[i]['L5']['I_noise']
        else:
            ext_inputs[i] = noise_inputs[i]

    simulator.run_step(step, ext_inputs)
    simulator.apply_plasticity(step * DT) # Pasar el tiempo actual

    if step % (N_STEPS_TRAIN // 10) == 0:
        print(f"  Progreso: {step / N_STEPS_TRAIN * 100:.0f}%")

final_weights = simulator.connection_manager.weights.copy()
print("Entrenamiento completado. Matriz de pesos estabilizada.")


# --- 3. Extracción de la Tabla de Verdad Emergente ---

def test_network_logic(trained_weights):
    """
    Testea la red con pesos congelados para extraer la tabla de verdad.
    """
    print("\n--- Extrayendo Tabla de Verdad Emergente ---")

    # Crear un nuevo simulador con los pesos entrenados y sin plasticidad
    rules_test = rules.copy()
    # Actualizar los pesos en las reglas (aunque el nuevo simulador los leerá de la matriz)
    # Esta parte es conceptual, el simulador usa la matriz directamente.

    columns_test = [CompartmentalColumn(index=i, n_nodes_per_layer={'L5': N_NEURONS},
                                        model_class=LIF_NodeGroup, model_params=lif_params)
                    for i in range(N_COLUMNS)]
    simulator_test = NetworkSimulator(columns_test, rules_test)
    simulator_test.connection_manager.weights = trained_weights # Cargar pesos entrenados

    truth_table = []

    # Iterar sobre las 4 combinaciones de entrada para C0 y C1
    for in_c0, in_c1 in [(0,0), (0,1), (1,0), (1,1)]:

        total_spikes_c2 = 0
        total_spikes_c3 = 0

        TEST_TIME_MS = 200
        N_STEPS_TEST = int(TEST_TIME_MS / DT)

        for step in range(N_STEPS_TEST):
            ext_inputs = {}
            if in_c0: ext_inputs[0] = {'L5': {'exc_spikes': stim_strength}}
            if in_c1: ext_inputs[1] = {'L5': {'exc_spikes': stim_strength}}

            simulator_test.run_step(step, ext_inputs)

            # Contar spikes en las columnas de salida
            total_spikes_c2 += np.sum(columns_test[2].layers['L5'].spikes)
            total_spikes_c3 += np.sum(columns_test[3].layers['L5'].spikes)

        # Determinar la salida binaria (dispara si la tasa supera un umbral)
        tasa_c2 = total_spikes_c2 / (N_STEPS_TEST * N_NEURONS)
        tasa_c3 = total_spikes_c3 / (N_STEPS_TEST * N_NEURONS)

        output_c2 = 1 if tasa_c2 > 0.01 else 0 # Umbral de tasa de disparo
        output_c3 = 1 if tasa_c3 > 0.01 else 0

        truth_table.append([in_c0, in_c1, output_c2, output_c3])
        print(f"  Input ({in_c0},{in_c1}) -> Output ({output_c2},{output_c3})")

    return pd.DataFrame(truth_table, columns=['Input_C0', 'Input_C1', 'Output_C2', 'Output_C3'])

# Ejecutar la extracción
truth_table_df = test_network_logic(final_weights)

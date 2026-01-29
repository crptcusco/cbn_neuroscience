# examples/learning_rules_comparison.py

import numpy as np
import matplotlib.pyplot as plt
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator
from cbn_neuroscience.core.plasticity_manager import PlasticityManager

# --- 1. Configuración Común ---
N_COLUMNS = 4
layer_names = ['L5'] # Capa única para simplificar
n_nodes_per_layer_rate = {'L5': 1}
n_nodes_per_layer_spikes = {'L5': 10}

# Conectividad total
rules = []
for i in range(N_COLUMNS):
    for j in range(N_COLUMNS):
        if i == j: continue
        rules.append({'sources': [(i, 'L5')], 'target_col': j, 'target_layer': 'L5',
                      'type': 'additive', 'weight': 0.5})

# --- 2. Configuración de la Red 1: STDP Multiplicativo ---
print("Configurando Red 1: STDP Multiplicativo (Modelo de Spikes)")
lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
              'R_m': 10.0, 'tau_syn_exc': 5.0, 'delta': 2.0, 'dt': 0.1}
stdp_params = {'a_plus': 0.1, 'a_minus': 0.1, 'tau_plus': 20.0, 'tau_minus': 20.0, 'w_max': 1.0}

columns_stdp = [CompartmentalColumn(index=i, n_nodes_per_layer=n_nodes_per_layer_spikes,
                                    model_class=LIF_NodeGroup, model_params=lif_params)
                for i in range(N_COLUMNS)]
plasticity_stdp = PlasticityManager(rule_type='stdp_multiplicative', **stdp_params)
simulator_stdp = NetworkSimulator(columns_stdp, rules, plasticity_stdp)


# --- 3. Configuración de la Red 2: Regla de Covarianza ---
print("Configurando Red 2: Regla de Covarianza (Modelo de Tasa)")
rate_params = {'tau_A': 20.0, 'gain_function_type': 'sigmoid', 'beta': 1.0, 'x0': 5.0}
cov_params = {'learning_rate': 0.1, 'w_max': 1.0, 'w_min': 0.0}

columns_cov = [CompartmentalColumn(index=i, n_nodes_per_layer=n_nodes_per_layer_rate,
                                 model_class=RateNodeGroup, model_params=rate_params)
               for i in range(N_COLUMNS)]
plasticity_cov = PlasticityManager(rule_type='covariance', **cov_params)
simulator_cov = NetworkSimulator(columns_cov, rules, plasticity_cov)


print("\nScripts de comparación configurados.")

# --- 4. Ejecución de las Simulaciones ---
SIM_TIME_MS = 1000
N_STEPS = int(SIM_TIME_MS / columns_stdp[0].layers['L5'].dt)

initial_weights_stdp = simulator_stdp.connection_manager.weights.copy()
initial_weights_cov = simulator_cov.connection_manager.weights.copy()

weight_history_stdp = []
weight_history_cov = []

print("\nEjecutando simulaciones...")
for step in range(N_STEPS):
    # Estímulo correlacionado: Col 0 y Col 2 reciben un input fuerte
    ext_inputs_stdp = {
        0: {'L5': {'exc_spikes': 2.0, 'I_noise': np.random.normal(0, 1, n_nodes_per_layer_spikes['L5'])}},
        2: {'L5': {'exc_spikes': 2.0, 'I_noise': np.random.normal(0, 1, n_nodes_per_layer_spikes['L5'])}}
    }
    ext_inputs_cov = {
        0: {'L5': {'I_noise': 10.0}},
        2: {'L5': {'I_noise': 10.0}}
    }

    # Simulación STDP
    simulator_stdp.run_step(step, ext_inputs_stdp)
    weight_history_stdp.append(simulator_stdp.connection_manager.get_weight(0, 'L5', 2, 'L5'))

    # Simulación Covarianza
    simulator_cov.run_step(step, ext_inputs_cov)
    weight_history_cov.append(simulator_cov.connection_manager.get_weight(0, 'L5', 2, 'L5'))

print("Simulaciones completadas.")

# --- 5. Visualización Comparativa ---
final_weights_stdp = simulator_stdp.connection_manager.weights.copy()
final_weights_cov = simulator_cov.connection_manager.weights.copy()

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

# Ax 1: Matriz Inicial
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(initial_weights_stdp, cmap='viridis', vmin=0, vmax=1.0)
ax1.set_title('Matriz de Pesos Inicial')

# Ax 2: Matriz Final STDP
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(final_weights_stdp, cmap='viridis', vmin=0, vmax=1.0)
ax2.set_title('Matriz Final (STDP Multiplicativo)')

# Ax 3: Matriz Final Covarianza
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(final_weights_cov, cmap='viridis', vmin=0, vmax=1.0)
ax3.set_title('Matriz Final (Regla de Covarianza)')

# Ax 4: Evolución de un peso
ax4 = fig.add_subplot(gs[1, 1])
time_axis = np.arange(N_STEPS) * columns_stdp[0].layers['L5'].dt
ax4.plot(time_axis, weight_history_stdp, label='Peso 0->2 (STDP)')
ax4.plot(time_axis, weight_history_cov, label='Peso 0->2 (Covarianza)')
ax4.set_title('Evolución del Peso de Conexión 0 -> 2')
ax4.set_xlabel('Tiempo (ms)')
ax4.set_ylabel('Peso (w)')
ax4.legend()
ax4.grid(True)

fig.colorbar(im, ax=[ax1, ax2, ax3])
plt.tight_layout()
plt.savefig('learning_rules_comparison.png')

print("\nGráfica comparativa guardada en 'learning_rules_comparison.png'")

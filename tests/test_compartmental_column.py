# tests/test_compartmental_column.py

import numpy as np
import pytest
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

def test_rule_based_propagation():
    """
    Valida que la propagación de señal funcione con la nueva
    arquitectura basada en reglas de acoplamiento.
    """
    # --- 1. Definición de la Red y Reglas ---
    lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
                  'R_m': 10.0, 'tau_syn_exc': 5.0, 'delta': 2.0, 'dt': 0.1}
    n_nodes = 5

    # Regla: Conexión aditiva simple de L4 a L5
    rules = [{'sources': [(0, 'L4')], 'target_col': 0, 'target_layer': 'L5',
              'type': 'additive', 'weight': 2.0}]

    column = CompartmentalColumn(index=0, n_nodes_per_layer={'L4': n_nodes, 'L5': n_nodes},
                                 lif_params=lif_params, coupling_rules=rules)

    # --- 2. Simulación ---
    N_STEPS = 500
    output_has_spiked = False
    network_state = {'col_0': {'L4': np.zeros(n_nodes, dtype=bool), 'L5': np.zeros(n_nodes, dtype=bool)}}

    for step in range(N_STEPS):
        ext_inputs = {}
        # Estimular L4 durante un tiempo
        if 100 < step < 300:
            ext_inputs = {'L4': {'additive': 2.0}}

        noise = {'L4': np.zeros(n_nodes), 'L5': np.zeros(n_nodes)}

        column.update(network_state=network_state, ext_inputs=ext_inputs, noise_inputs=noise)

        # Actualizar el estado de la red para el siguiente ciclo
        network_state['col_0'] = column.get_layer_spikes()

        if np.any(column.get_layer_spikes()['L5']):
            output_has_spiked = True
            break

    # --- 3. Verificación ---
    assert output_has_spiked, "La señal no se propagó de L4 a L5 con el motor de reglas."

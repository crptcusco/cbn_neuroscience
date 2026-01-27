# tests/test_compartmental_column.py

import numpy as np
import pytest
from cbn_neuroscience.core.lif_nodegroup import LIF_NodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

def test_lif_propagation_with_simulator():
    """
    Valida la propagación de señal en un modelo LIF a través
    de la nueva arquitectura con NetworkSimulator.
    """
    lif_params = {'tau_m': 15.0, 'theta': -55.0, 'v_reset': -70.0, 'v_rest': -70.0,
                  'R_m': 10.0, 'tau_syn_exc': 5.0, 'delta': 2.0, 'dt': 0.1}
    n_nodes = 5

    rules = [{'sources': [(0, 'L4')], 'target_col': 0, 'target_layer': 'L5',
              'type': 'additive', 'weight': 2.0}]

    columns = [CompartmentalColumn(index=0, n_nodes_per_layer={'L4': n_nodes, 'L5': n_nodes},
                                   model_class=LIF_NodeGroup, model_params=lif_params)]

    simulator = NetworkSimulator(columns, coupling_rules=rules)

    output_has_spiked = False
    for i in range(500):
        ext_inputs = {0: {'L4': {'exc_spikes': 2.0}}} # Estímulo constante
        simulator.run_step(i, ext_inputs)

        # Comprobar si la capa de salida ha disparado
        if np.any(columns[0].layers['L5'].spikes):
            output_has_spiked = True
            break

    assert output_has_spiked, "La señal no se propagó de L4 a L5 con la nueva arquitectura."

# tests/test_nonlinear_logic_gates.py

import numpy as np
import pytest
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

def test_and_gate_logic_with_simulator():
    """Valida una compuerta AND con la nueva arquitectura."""
    rate_params = {'tau_A': 5.0, 'gain_function_type': 'sigmoid', 'beta': 5.0, 'x0': 0.8}
    n_nodes_per_layer = {'IN_A': 1, 'IN_B': 1, 'OUT': 1}

    rules = [{'sources': [(0, 'IN_A'), (0, 'IN_B')], 'target_col': 0, 'target_layer': 'OUT',
              'type': 'multiplicative', 'weight': 5.0}]

    columns = [CompartmentalColumn(index=0, n_nodes_per_layer=n_nodes_per_layer,
                                   model_class=RateNodeGroup, model_params=rate_params)]

    simulator = NetworkSimulator(columns, coupling_rules=rules)

    def run_scenario(input_a, input_b):
        # Resetear el estado del simulador
        simulator.network_state = {f'col_{i}': col.get_state() for i, col in enumerate(simulator.columns)}
        for layer in simulator.columns[0].layers.values(): layer.A.fill(0)

        ext_inputs = {0: {'IN_A': {'I_total': input_a}, 'IN_B': {'I_total': input_b}}}
        for _ in range(100):
            simulator.run_step(ext_inputs)
        return simulator.network_state['col_0']['OUT'][0]

    assert run_scenario(0, 0) < 0.1
    assert run_scenario(1, 0) < 0.1
    assert run_scenario(0, 1) < 0.1
    assert run_scenario(1, 1) > 0.5

# tests/test_network_simulator.py

import numpy as np
import pytest
from cbn_neuroscience.core.rate_nodegroup import RateNodeGroup
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn
from cbn_neuroscience.core.network_simulator import NetworkSimulator

def test_connection_manager_initialization():
    """
    Valida que el ConnectionManager dentro del NetworkSimulator
    inicializa correctamente la matriz de pesos a partir de las reglas.
    """
    rate_params = {'tau_A': 10.0}
    columns = [
        CompartmentalColumn(index=0, n_nodes_per_layer={'L5': 1}, model_class=RateNodeGroup, model_params=rate_params),
        CompartmentalColumn(index=1, n_nodes_per_layer={'L5': 1}, model_class=RateNodeGroup, model_params=rate_params)
    ]

    rules = [
        {'sources': [(0, 'L5')], 'target_col': 1, 'target_layer': 'L5', 'weight': 0.8}
    ]

    simulator = NetworkSimulator(columns, rules)

    # El ConnectionManager debería haber creado una matriz de pesos 2x2
    # (una capa por columna)
    assert simulator.connection_manager.weights.shape == (2, 2)

    # El peso de la conexión de Col 0 -> Col 1 debería ser 0.8
    weight = simulator.connection_manager.get_weight(0, 'L5', 1, 'L5')
    assert np.isclose(weight, 0.8)

    # El peso de la conexión de Col 1 -> Col 0 (no definida) debería ser 0
    weight_undefined = simulator.connection_manager.get_weight(1, 'L5', 0, 'L5')
    assert np.isclose(weight_undefined, 0.0)

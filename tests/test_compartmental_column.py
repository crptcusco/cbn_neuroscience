# tests/test_compartmental_column.py

import numpy as np
import pytest
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

def test_glif_synaptic_persistence_propagation():
    """
    Prueba con el modelo GLIF: Un tren de spikes en L4 debe generar una
    conductancia persistente que active la capa de salida L5/6.
    """
    # 1. Configurar la columna con el modelo GLIF
    layer_sizes = {'L4': 5, 'L2/3': 5, 'L5/6': 5}
    lif_params = {
        'tau_m': 15.0, 'theta': -55.0, 'v_rest': -70.0, 'R_m': 10.0,
        'tau_syn_exc': 5.0, 'E_exc': 0.0, 'delta': 2.0, 'dt': 0.1
    }
    column = CompartmentalColumn(
        index=0,
        n_nodes_per_layer=layer_sizes,
        g_axial=4.0,  # Peso de la conexión inter-capa (incremento de conductancia)
        lif_params=lif_params
    )

    # 2. Definir un tren de spikes de entrada y sin ruido
    stimulus_spike_weight = 0.5 # Incremento de g_exc por spike
    stimulus_duration_steps = 200 # 20 ms

    total_nodes = sum(layer_sizes.values())
    noise_input = np.zeros(total_nodes) # Sin ruido
    inter_column_input = np.zeros(layer_sizes['L2/3'])

    # 3. Simular la columna
    simulation_steps = 500 # 50 ms
    output_has_spiked = False

    for step in range(simulation_steps):
        # Enviar un pulso de spikes a L4
        ext_spikes = np.full(layer_sizes['L4'], stimulus_spike_weight) if step < stimulus_duration_steps else np.zeros(layer_sizes['L4'])

        column.update(
            ext_spikes=ext_spikes,
            I_noise_total=noise_input,
            inter_column_spikes=inter_column_input
        )

        if np.any(column.layers['L5/6'].spikes):
            output_has_spiked = True
            break

    # 4. Verificación
    assert output_has_spiked, (
        "La capa de salida NO disparó con el modelo GLIF. "
        "La persistencia sináptica no fue efectiva."
    )

    assert step > 1, "La capa de salida disparó demasiado rápido."

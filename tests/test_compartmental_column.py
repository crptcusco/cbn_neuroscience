# tests/test_compartmental_column.py

import numpy as np
import pytest
from cbn_neuroscience.core.compartmental_column import CompartmentalColumn

def test_compartmental_column_litmus_test():
    """
    Prueba de Fuego: Un estímulo en la capa de entrada (L4) debe
    propagarse a través del acoplamiento axial y activar la capa de salida (L5/6).
    """
    # 1. Configurar la columna
    layer_sizes = {'L4': 5, 'L2/3': 5, 'L5/6': 5}
    column = CompartmentalColumn(index=0, n_nodes_per_layer=layer_sizes, g_axial=0.8)

    # 2. Definir el estímulo
    # Una corriente fuerte aplicada solo a la capa de entrada (L4)
    # y solo durante unos pocos pasos de tiempo para simular un pulso.
    stimulus_duration = 10  # en pasos de simulación
    I_ext = np.full(layer_sizes['L4'], 2.0)

    # 3. Simular la columna
    # Simular durante suficientes pasos para que la señal se propague.
    simulation_steps = 100
    output_has_spiked = False

    for step in range(simulation_steps):
        current_stimulus = I_ext if step < stimulus_duration else np.zeros(layer_sizes['L4'])
        column.update(I_ext=current_stimulus)

        # Comprobar si la capa de salida ha disparado
        if np.any(column.layers['L5/6'].states == 1):
            output_has_spiked = True
            break

    # 4. Verificación
    assert output_has_spiked, (
        "La capa de salida (L5/6) nunca se activó. "
        "El acoplamiento compartimental interno falló en propagar la señal."
    )

    # Verificación adicional: la capa de salida no debería activarse inmediatamente
    assert step > 1, "La capa de salida se activó demasiado pronto, sugiere un error."

# sanity_check.py

import numpy as np
import random

from cbn_neuroscience.core.laminar_template import LaminarColumnTemplate
from cbn_neuroscience.core.factory import generate_laminar_cbn
from cbn_neuroscience.core.simulation import Simulator
from cbn_neuroscience.core.anticipation_engine import AnticipationMonitor

def run_sanity_check():
    """
    Ejecuta una simulación simple para verificar que los componentes
    centrales de cbn_neuroscience funcionan juntos.
    """
    print("--- [INICIANDO SANITY CHECK REFACTORIZADO] ---")
    random.seed(42)
    np.random.seed(42)

    # 1. Configurar y crear una única columna laminar
    print("1. Creando una columna laminar...")
    layer_sizes = {'L2/3': 10, 'L4': 5, 'L5/6': 8}
    template = LaminarColumnTemplate(
        layer_sizes=layer_sizes,
        n_input_variables=2,
        n_output_variables=2
    )

    cbn = generate_laminar_cbn(template=template, n_local_networks=1)
    column = cbn.l_local_networks[0]

    print(f"   - Columna creada con {len(column.internal_variables)} neuronas.")

    # 2. Configurar el simulador y el monitor
    print("\n2. Configurando el simulador y el monitor...")
    simulator = Simulator(column)
    monitor = AnticipationMonitor()

    # 3. Ejecutar la simulación
    print("\n3. Ejecutando simulación...")
    initial_state = np.random.randint(0, 2, size=len(column.internal_variables))
    column.current_state = initial_state

    p_error = 0.01
    max_steps = 100

    print(f"   - Estado inicial: {initial_state}")

    # Bucle de simulación principal
    for step in range(max_steps):
        # El simulador calcula el siguiente estado determinista
        # Para esta prueba simple, no hay entradas externas.
        next_deterministic_state, _ = simulator.run(column.current_state, steps=1)

        # La columna aplica la incertidumbre y actualiza su propio estado
        current_state_arr = column.update_with_uncertainty(next_deterministic_state[1], p_error)

        # El monitor comprueba la estabilidad
        is_stable, message = monitor.check_stability(current_state_arr)

        print(f"   Paso {step+1}: Estado -> {current_state_arr} | {message}")

        if is_stable:
            print("\n   ¡Estabilidad alcanzada!")
            break

    # 4. Medir la densidad de actividad
    print("\n4. Analizando el estado final...")
    final_density = column.get_sparse_density()
    print(f"   - Densidad de actividad (Sparsity) del estado final: {final_density:.2f}")

    print("\n--- [SANITY CHECK COMPLETADO] ---")

if __name__ == "__main__":
    run_sanity_check()

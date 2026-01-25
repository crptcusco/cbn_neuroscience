# tests/test_simulation.py

import unittest
import numpy as np
from cbnetwork.localnetwork import LocalNetwork
from cbnetwork.internalvariable import InternalVariable
from cbn_neuroscience.core.simulation import Simulator

class TestSimulation(unittest.TestCase):

    def setUp(self):
        """Prepara una red de prueba simple para la simulación."""
        # Crear una red de 3 nodos:
        # Nodo 1: Se activa si el Nodo 2 está activo (función: 2)
        # Nodo 2: Se activa si el Nodo 3 está activo (función: 3)
        # Nodo 3: Se activa si el Nodo 1 está activo (función: 1)
        # Esto crea un oscilador de 3 pasos.
        var1 = InternalVariable(index=1, cnf_function=[[2]])
        var2 = InternalVariable(index=2, cnf_function=[[3]])
        var3 = InternalVariable(index=3, cnf_function=[[1]])

        self.network = LocalNetwork(index=0, internal_variables=[1, 2, 3])
        self.network.descriptive_function_variables = [var1, var2, var3]

        self.simulator = Simulator(self.network)

    def test_simulation_run(self):
        """Prueba una ejecución de simulación simple para verificar la transición de estado."""
        # Estado inicial: solo el nodo 1 está activo
        initial_state = np.array([1, 0, 0])
        steps = 3

        history, _ = self.simulator.run(initial_state, steps)

        # El historial esperado:
        # Paso 0: [1, 0, 0] (inicial)
        # Paso 1: [0, 0, 1] (Nodo 3 se activa por el Nodo 1)
        # Paso 2: [0, 1, 0] (Nodo 2 se activa por el Nodo 3)
        # Paso 3: [1, 0, 0] (Nodo 1 se activa por el Nodo 2)

        expected_history = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])

        np.testing.assert_array_equal(history, expected_history)

    def test_modulation(self):
        """Prueba que el simulador puede manejar reglas lógicas dinámicas (modulación)."""
        # Crear una red de 2 nodos:
        # Nodo 1: Siempre es 1
        # Nodo 2: Su lógica depende de una entrada externa `mod_signal`.
        # Si `mod_signal` es 1, la función es `1` (activado).
        # Si `mod_signal` es 0, la función es `-1` (desactivado).

        def modulated_function(external_values):
            if external_values.get('mod_signal') == 1:
                return [[1]]  # Se activa si el nodo 1 está activo
            else:
                return [[-1]] # Se desactiva si el nodo 1 está inactivo

        var1_mod = InternalVariable(index=1, cnf_function=[[1]]) # Se activa a sí mismo (estable)
        var2_mod = InternalVariable(index=2, cnf_function=modulated_function)

        network_mod = LocalNetwork(index=1, internal_variables=[1, 2])
        network_mod.descriptive_function_variables = [var1_mod, var2_mod]

        simulator_mod = Simulator(network_mod)

        initial_state = np.array([1, 0])
        steps = 2

        # Simulación 1: la señal del modulador está activada
        external_inputs_on = [{'mod_signal': 1}, {'mod_signal': 1}]
        history_on, _ = simulator_mod.run(initial_state, steps, external_inputs=external_inputs_on)
        # El nodo 2 debería activarse
        self.assertEqual(history_on[-1, 1], 1)

        # Simulación 2: la señal del modulador está desactivada
        external_inputs_off = [{'mod_signal': 0}, {'mod_signal': 0}]
        history_off, _ = simulator_mod.run(initial_state, steps, external_inputs=external_inputs_off)
        # El nodo 2 debería permanecer desactivado
        self.assertEqual(history_off[-1, 1], 0)


if __name__ == '__main__':
    unittest.main()

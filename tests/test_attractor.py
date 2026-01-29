# tests/test_attractor.py

import unittest
import numpy as np
from cbnetwork.localnetwork import LocalNetwork
from cbnetwork.internalvariable import InternalVariable
from cbn_neuroscience.core.simulation import Simulator
from cbn_neuroscience.core.anticipation_engine import AnticipationMonitor

class TestAttractorDetection(unittest.TestCase):

    def test_fixed_point_attractor(self):
        """Prueba la detección de un atractor de punto fijo."""
        var1 = InternalVariable(index=1, cnf_function=[[1]])
        var2 = InternalVariable(index=2, cnf_function=[[2], [-2]])

        network = LocalNetwork(index=0, internal_variables=[1, 2])
        network.descriptive_function_variables = [var1, var2]

        simulator = Simulator(network)
        monitor = AnticipationMonitor()

        current_state = np.array([1, 0])
        monitor.check_stability(tuple(current_state)) # Registrar estado inicial

        # Simular un paso
        history, _ = simulator.run(current_state, steps=1)
        next_state = history[1]

        is_stable, info = monitor.check_stability(tuple(next_state))

        self.assertTrue(is_stable)
        self.assertIn("Punto Fijo", info)
        np.testing.assert_array_equal(current_state, next_state)

    def test_limit_cycle_attractor(self):
        """Prueba la detección de un atractor de ciclo límite."""
        var1 = InternalVariable(index=1, cnf_function=[[2]])
        var2 = InternalVariable(index=2, cnf_function=[[3]])
        var3 = InternalVariable(index=3, cnf_function=[[1]])

        network = LocalNetwork(index=1, internal_variables=[1, 2, 3])
        network.descriptive_function_variables = [var1, var2, var3]

        simulator = Simulator(network)
        monitor = AnticipationMonitor()

        current_state = np.array([1, 0, 0])
        monitor.check_stability(tuple(current_state))

        # Simular 3 pasos para completar el ciclo
        for _ in range(3):
            history, _ = simulator.run(current_state, steps=1)
            current_state = history[1]
            is_stable, info = monitor.check_stability(tuple(current_state))

        self.assertTrue(is_stable)
        self.assertIn("Ciclo Límite de longitud 3", info)

    def test_no_attractor_found(self):
        """Prueba el comportamiento cuando no se encuentra un atractor en los pasos dados."""
        var1 = InternalVariable(index=1, cnf_function=[[2]])
        var2 = InternalVariable(index=2, cnf_function=[[1]]) # Oscilador simple

        network = LocalNetwork(index=2, internal_variables=[1, 2])
        network.descriptive_function_variables = [var1, var2]

        simulator = Simulator(network)
        monitor = AnticipationMonitor()

        current_state = np.array([1, 0])
        monitor.check_stability(tuple(current_state))

        # Simular solo 1 paso, no es suficiente para detectar el ciclo
        history, _ = simulator.run(current_state, steps=1)
        next_state = history[1]

        is_stable, _ = monitor.check_stability(tuple(next_state))

        self.assertFalse(is_stable)

if __name__ == '__main__':
    unittest.main()

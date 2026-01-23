# tests/test_attractor.py

import unittest
import numpy as np
from cbnetwork.localnetwork import LocalNetwork
from cbnetwork.internalvariable import InternalVariable
from cbn_neuroscience.core.simulation import Simulator
from cbn_neuroscience.core.anticipation_engine import AnticipationMonitor

class TestAttractorDetection(unittest.TestCase):

    def test_fixed_point_attractor(self):
        """Prueba la detección de un atractor de punto fijo exacto."""
        var1 = InternalVariable(index=1, cnf_function=[[1]])
        var2 = InternalVariable(index=2, cnf_function=[[2], [-2]])

        network = LocalNetwork(index=0, internal_variables=[1, 2])
        network.descriptive_function_variables = [var1, var2]

        simulator = Simulator(network)
        monitor = AnticipationMonitor(threshold=0.001) # Umbral pequeño para igualdad estricta

        current_state = np.array([1, 0])
        monitor.check_stability(current_state)

        history, _ = simulator.run(current_state, steps=1)
        next_state = history[1]

        is_stable, info = monitor.check_stability(next_state)

        self.assertTrue(is_stable)
        self.assertIn("Punto Fijo", info)
        np.testing.assert_array_equal(current_state, next_state)

    def test_limit_cycle_attractor(self):
        """Prueba la detección de un atractor de ciclo límite exacto."""
        var1 = InternalVariable(index=1, cnf_function=[[2]])
        var2 = InternalVariable(index=2, cnf_function=[[3]])
        var3 = InternalVariable(index=3, cnf_function=[[1]])

        network = LocalNetwork(index=1, internal_variables=[1, 2, 3])
        network.descriptive_function_variables = [var1, var2, var3]

        simulator = Simulator(network)
        monitor = AnticipationMonitor(threshold=0.001)

        current_state = np.array([1, 0, 0])
        monitor.check_stability(current_state)

        is_stable = False
        info = ""
        for _ in range(3):
            history, _ = simulator.run(current_state, steps=1)
            current_state = history[1]
            is_stable, info = monitor.check_stability(current_state)

        self.assertTrue(is_stable)
        self.assertIn("Ciclo Límite", info)

    def test_no_attractor_found(self):
        """Prueba que no se detecta estabilidad si la distancia está por encima del umbral."""
        var1 = InternalVariable(index=1, cnf_function=[[2]])
        var2 = InternalVariable(index=2, cnf_function=[[1]])

        network = LocalNetwork(index=2, internal_variables=[1, 2])
        network.descriptive_function_variables = [var1, var2]

        simulator = Simulator(network)
        monitor = AnticipationMonitor(threshold=0.001)

        current_state = np.array([1, 0])
        monitor.check_stability(current_state)

        history, _ = simulator.run(current_state, steps=1)
        next_state = history[1]

        is_stable, _ = monitor.check_stability(next_state)

        self.assertFalse(is_stable)

    def test_noisy_convergence(self):
        """Prueba que se detecta convergencia incluso con ruido (usando Distancia de Hamming)."""
        monitor = AnticipationMonitor(threshold=0.1) # Umbral del 10%

        # Estado base de 20 bits
        base_state = np.zeros(20, dtype=int)
        monitor.check_stability(base_state)

        # Crear un estado "ruidoso" que difiere en 1 bit (5% de distancia)
        noisy_state = base_state.copy()
        noisy_state[0] = 1 # Invertir un bit

        is_stable, info = monitor.check_stability(noisy_state)

        self.assertTrue(is_stable)
        self.assertIn("Punto Fijo", info)

if __name__ == '__main__':
    unittest.main()

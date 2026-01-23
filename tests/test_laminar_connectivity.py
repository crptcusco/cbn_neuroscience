# tests/test_laminar_connectivity.py

import unittest
import random
from cbn_neuroscience.core.laminar_template import LaminarColumnTemplate

class TestLaminarConnectivityAndValidation(unittest.TestCase):

    def setUp(self):
        """Prepara una configuración base para las pruebas."""
        self.layer_sizes = {'L2/3': 20, 'L4': 5, 'L5/6': 10}
        self.total_vars = sum(self.layer_sizes.values())

    def test_input_validation_raises_error(self):
        """Prueba que se lanza un ValueError si hay demasiadas variables de entrada."""
        with self.assertRaises(ValueError):
            LaminarColumnTemplate(
                layer_sizes=self.layer_sizes,
                n_input_variables=10,  # Más que el tamaño de la capa L4 (5)
                n_output_variables=5
            )

    def test_output_validation_raises_error(self):
        """Prueba que se lanza un ValueError si hay demasiadas variables de salida."""
        with self.assertRaises(ValueError):
            LaminarColumnTemplate(
                layer_sizes=self.layer_sizes,
                n_input_variables=3,
                n_output_variables=15  # Más que el tamaño de la capa L5/6 (10)
            )

    def test_connectivity_bias(self):
        """
        Prueba estadísticamente que los nodos de las capas II/III se conectan
        preferentemente entre sí.
        """
        # Usar una semilla para la reproducibilidad
        random.seed(42)

        # Crear una plantilla con un sesgo de conectividad muy alto para que la señal sea clara
        template = LaminarColumnTemplate(
            layer_sizes=self.layer_sizes,
            n_input_variables=2,
            n_output_variables=2,
            connectivity_bias=10.0  # Sesgo alto
        )

        associative_nodes = set(template.layers['L2/3'])

        # Contar las conexiones desde los nodos de la capa II/III
        intra_associative_connections = 0
        total_connections_from_associative = 0

        for var_index, cnf in template.d_variable_cnf_function.items():
            if var_index in associative_nodes:
                for clause in cnf:
                    for literal in clause:
                        # Ignorar las conexiones de auto-referencia y las externas
                        literal_var = abs(literal)
                        if literal_var != var_index and literal_var <= self.total_vars:
                            total_connections_from_associative += 1
                            if literal_var in associative_nodes:
                                intra_associative_connections += 1

        # Calcular la proporción de conexiones intra-asociativas
        if total_connections_from_associative > 0:
            bias_ratio = intra_associative_connections / total_connections_from_associative
        else:
            bias_ratio = 0

        # Calcular la proporción esperada sin sesgo.
        # Es la probabilidad de elegir un nodo de la capa asociativa del total de nodos.
        expected_ratio_no_bias = len(associative_nodes) / self.total_vars

        # Verificar que la proporción observada es significativamente mayor que la esperada sin sesgo.
        # Un umbral razonable es que el sesgo duplique la probabilidad.
        self.assertGreater(bias_ratio, expected_ratio_no_bias * 1.5,
                           f"La proporción de sesgo ({bias_ratio:.2f}) no fue significativamente mayor "
                           f"que la esperada sin sesgo ({expected_ratio_no_bias:.2f}).")

if __name__ == '__main__':
    unittest.main()

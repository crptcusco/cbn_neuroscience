# tests/test_coupling.py

import unittest
from cbn_neuroscience.core.laminar_template import LaminarColumnTemplate
from cbn_neuroscience.core.factory import generate_laminar_cbn

class TestCoupling(unittest.TestCase):

    def test_multi_column_creation(self):
        """
        Prueba que la fábrica puede crear una red de múltiples columnas
        acopladas sin errores.
        """
        # 1. Configurar la plantilla
        layer_sizes = {'L2/3': 5, 'L4': 3, 'L5/6': 4}
        template = LaminarColumnTemplate(
            layer_sizes=layer_sizes,
            n_input_variables=2,
            n_output_variables=2
        )

        # 2. Generar la red con 2 columnas en una topología completa
        n_columns = 2
        try:
            cbn = generate_laminar_cbn(
                template=template,
                n_local_networks=n_columns,
                v_topology=1  # 1 = Grafo Completo
            )
        except Exception as e:
            self.fail(f"generate_laminar_cbn levantó una excepción inesperada: {e}")

        # 3. Verificar la estructura de la red
        # ¿Se crearon el número correcto de columnas?
        self.assertEqual(len(cbn.l_local_networks), n_columns)

        # ¿Se creó el acoplamiento? En un grafo completo de 2 nodos, debe haber 2 ejes.
        self.assertEqual(len(cbn.l_directed_edges), 2)

        # ¿Las redes locales tienen señales de entrada asignadas?
        for network in cbn.l_local_networks:
            self.assertTrue(len(network.input_signals) > 0)

if __name__ == '__main__':
    unittest.main()

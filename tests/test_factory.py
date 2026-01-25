# tests/test_factory.py

import unittest
from cbn_neuroscience.core.laminar_template import LaminarColumnTemplate
from cbn_neuroscience.core.laminar_column import LaminarColumn
from cbn_neuroscience.core.factory import generate_laminar_cbn

class TestFactory(unittest.TestCase):

    def test_generate_laminar_cbn(self):
        """
        Prueba que la función de fábrica crea correctamente una CBN
        con redes locales del tipo LaminarColumn.
        """
        # 1. Configurar la plantilla
        layer_sizes = {'L2/3': 4, 'L4': 2, 'L5/6': 3}
        template = LaminarColumnTemplate(
            layer_sizes=layer_sizes,
            n_input_variables=2,
            n_output_variables=2
        )

        # 2. Generar la red
        n_local_networks = 3
        cbn = generate_laminar_cbn(
            template=template,
            n_local_networks=n_local_networks
        )

        # 3. Verificar la estructura
        self.assertEqual(len(cbn.l_local_networks), n_local_networks)

        # 4. Verificar el tipo y contenido de cada red local
        for network in cbn.l_local_networks:
            # La prueba más importante: ¿es del tipo correcto?
            self.assertIsInstance(network, LaminarColumn)

            # ¿Contiene la información de las capas?
            self.assertTrue(hasattr(network, 'layer_map'))
            self.assertIsNotNone(network.layer_map)

            # ¿Son correctas las capas?
            self.assertEqual(network.layer_map.keys(), layer_sizes.keys())

            # Verificar que los nodos en las capas de la red instanciada
            # coinciden con los nodos definidos en la plantilla.
            # Esto confirma que la información de las capas se transfirió correctamente.
            template_layer_nodes = set(n for l in template.layers.values() for n in l)
            network_internal_vars = set(network.internal_variables)

            # Ajustar los nodos de la plantilla para que coincidan con el rango de esta red
            min_var = min(network_internal_vars)
            expected_nodes_in_network = {n + min_var - 1 for n in template_layer_nodes}

            self.assertEqual(network_internal_vars, expected_nodes_in_network)


if __name__ == '__main__':
    unittest.main()

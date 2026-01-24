# cbn_neuroscience/laminar.py

import random
from cbnetwork.localtemplates import LocalNetworkTemplate
from cbnetwork.localnetwork import LocalNetwork
from cbnetwork.cnflist import CNFList

from .laminar_column import LaminarColumn

class LaminarColumnTemplate(LocalNetworkTemplate):
    """
    Una plantilla para crear redes locales con una estructura laminar predefinida,
    con conectividad de alta densidad en las capas asociativas.
    """
    def __init__(self, layer_sizes: dict, connectivity_bias: float = 2.0, **kwargs):
        self.layer_sizes = layer_sizes
        self.connectivity_bias = connectivity_bias
        n_vars_network = sum(layer_sizes.values())
        super().__init__(n_vars_network=n_vars_network, **kwargs)

    def generate_local_dynamic(self):
        """
        Genera la dinámica local para una red laminar, con validación y
        conectividad de alta densidad para las capas II/III.
        """
        if self.n_input_variables > self.layer_sizes.get('L4', 0):
            raise ValueError("El número de variables de entrada no puede exceder el tamaño de la Capa L4.")
        if self.n_output_variables > self.layer_sizes.get('L5/6', 0):
            raise ValueError("El número de variables de salida no puede exceder el tamaño de la Capa L5/6.")

        start_index = 1
        l_internal_var_indexes = list(range(start_index, self.n_vars_network + start_index))
        self.layers = {}
        current_pos = 0
        for layer, size in self.layer_sizes.items():
            self.layers[layer] = l_internal_var_indexes[current_pos : current_pos + size]
            current_pos += size

        input_signal_start_index = self.n_vars_network + 1
        l_input_coupling_signal_indexes = list(range(input_signal_start_index, input_signal_start_index + self.n_input_variables))
        nodes_for_input = random.sample(self.layers['L4'], self.n_input_variables)
        self.l_output_var_indexes = random.sample(self.layers['L5/6'], self.n_output_variables)

        self.d_variable_cnf_function = {}
        associative_layers = self.layers.get('L2/3', [])

        for i_variable in l_internal_var_indexes:
            input_coup_sig_index = None
            if i_variable in nodes_for_input:
                input_coup_sig_index = random.choice(l_input_coupling_signal_indexes)
                l_input_coupling_signal_indexes.remove(input_coup_sig_index)

            if i_variable in associative_layers:
                preferred_pool = associative_layers * int(self.connectivity_bias)
                other_pool = [n for n in l_internal_var_indexes if n not in associative_layers]
                biased_vars_pool = preferred_pool + other_pool
            else:
                biased_vars_pool = l_internal_var_indexes

            self.d_variable_cnf_function[i_variable] = CNFList.generate_cnf(
                l_inter_vars=biased_vars_pool,
                input_coup_sig_index=input_coup_sig_index,
                max_clauses=self.n_max_of_clauses,
                max_literals=self.n_max_of_literals,
            )

    def create_network(self, index: int, internal_variables: list) -> LaminarColumn:
        """
        Método de fábrica para crear una instancia de LaminarColumn.

        Args:
            index (int): El índice de la red local.
            internal_variables (list): La lista de variables internas para la red.

        Returns:
            LaminarColumn: Una nueva instancia de la columna laminar.
        """
        # Aquí se transfiere la información de las capas a la instancia de la columna
        return LaminarColumn(
            index=index,
            internal_variables=internal_variables,
            layer_map=self.layers
        )

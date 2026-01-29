# cbn_neuroscience/factory.py

from cbnetwork.cbnetwork import CBN
from cbnetwork.globaltopology import GlobalTopology
from cbnetwork.coupling import OrCoupling
from cbnetwork.internalvariable import InternalVariable
from cbnetwork.directededge import DirectedEdge
from .laminar_template import LaminarColumnTemplate

def generate_laminar_cbn(
    template: LaminarColumnTemplate,
    n_local_networks: int,
    v_topology: int = 1,
    coupling_strategy=None
) -> CBN:
    """
    Crea una Red Booleana Acoplada (CBN) utilizando una plantilla laminar.
    """
    if coupling_strategy is None:
        coupling_strategy = OrCoupling()

    # 1. Generar la topología global
    o_global_topology = GlobalTopology.generate_sample_topology(
        v_topology=v_topology, n_nodes=n_local_networks
    )

    # 2. Generar las redes locales
    l_local_networks = []
    variable_count = 1
    for i in range(1, n_local_networks + 1):
        internal_vars = list(range(variable_count, variable_count + template.n_vars_network))
        o_local_network = template.create_network(
            index=i, internal_variables=internal_vars
        )
        l_local_networks.append(o_local_network)
        variable_count += template.n_vars_network

    # 3. Generar los ejes dirigidos (lógica corregida)
    l_directed_edges = []
    i_last_variable = l_local_networks[-1].internal_variables[-1] + 1
    i_directed_edge = 1
    # Crear un mapa de índice a red para una búsqueda rápida
    network_map = {net.index: net for net in l_local_networks}

    for relation in o_global_topology.l_edges:
        output_ln_idx, input_ln_idx = relation

        # Lógica correcta para obtener las variables de salida
        output_network = network_map[output_ln_idx]
        output_indices_in_template = template.l_output_var_indexes
        # Mapear los índices de la plantilla a los índices reales de la red de salida
        output_variables = [
            var for i, var in enumerate(output_network.internal_variables)
            if (i + 1) in output_indices_in_template
        ]

        coupling_function = coupling_strategy.generate_coupling_function(output_variables)

        o_directed_edge = DirectedEdge(
            index=i_directed_edge,
            index_variable_signal=i_last_variable,
            input_local_network=input_ln_idx,
            output_local_network=output_ln_idx,
            l_output_variables=output_variables,
            coupling_function=coupling_function
        )
        l_directed_edges.append(o_directed_edge)
        i_last_variable += 1
        i_directed_edge += 1

    # 4. Procesar las señales de entrada para cada red
    for o_local_network in l_local_networks:
        input_signals = [edge for edge in l_directed_edges if edge.input_local_network == o_local_network.index]
        o_local_network.process_input_signals(input_signals=input_signals)

    # 5. Generar la dinámica local
    for o_local_network in l_local_networks:
        for i_local_variable in o_local_network.internal_variables:
            template_var_index = (i_local_variable - 1) % template.n_vars_network + 1
            cnf_function = template.d_variable_cnf_function.get(template_var_index, [])
            o_internal_variable = InternalVariable(
                index=i_local_variable, cnf_function=cnf_function
            )
            o_local_network.descriptive_function_variables.append(o_internal_variable)

    # 6. Crear la instancia final de CBN
    o_cbn = CBN(
        l_local_networks=l_local_networks, l_directed_edges=l_directed_edges
    )
    o_cbn.o_global_topology = o_global_topology

    return o_cbn

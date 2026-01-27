# cbn_neuroscience/core/network_simulator.py

import numpy as np
from cbn_neuroscience.core.connections import ConnectionManager

class NetworkSimulator:
    """
    Orquesta la simulación, usando un ConnectionManager para calcular las interacciones.
    """
    def __init__(self, columns, coupling_rules):
        self.columns = columns
        self.connection_manager = ConnectionManager(columns, coupling_rules)
        self.network_state = {f'col_{i}': col.get_state() for i, col in enumerate(self.columns)}

    def run_step(self, ext_inputs: dict):
        """
        Calcula todos los inputs y actualiza cada columna.
        """
        # --- 1. Calcular inputs de acoplamiento para todas las capas ---
        all_coupling_inputs = {i: {name: {} for name in col.layers.keys()} for i, col in enumerate(self.columns)}

        for rule in self.connection_manager.coupling_rules:
            target_col_idx = rule['target_col']
            target_layer = rule['target_layer']

            source_values = [self.network_state[f'col_{c}'][l] for c, l in rule['sources']]
            is_spike_based = isinstance(source_values[0], np.ndarray) and source_values[0].dtype == bool

            # Para modelos de spikes, la actividad es la tasa de disparo media
            if is_spike_based:
                source_values = [np.mean(s) for s in source_values]

            # Obtener peso dinámico
            weight = self.connection_manager.get_weight(rule['sources'][0][0], rule['sources'][0][1], target_col_idx, target_layer)

            # Calcular valor de la interacción
            value = np.prod(source_values) if rule['type'] == 'multiplicative' else np.sum(source_values)

            # Determinar el tipo de input y acumular
            input_type = rule.get('synapse_type', 'I_total' if not is_spike_based else 'weighted_spikes')

            current_inputs = all_coupling_inputs[target_col_idx][target_layer]
            current_inputs.setdefault(input_type, 0)
            current_inputs[input_type] += weight * value

        # --- 2. Actualizar cada columna ---
        for i, col in enumerate(self.columns):
            # Combinar inputs externos y de acoplamiento
            final_inputs = {name: {} for name in col.layers.keys()}
            col_ext_inputs = ext_inputs.get(i, {})
            col_coupling_inputs = all_coupling_inputs[i]

            for layer_name in final_inputs.keys():
                final_inputs[layer_name] = col_coupling_inputs.get(layer_name, {})
                # Sumar inputs externos
                for in_type, in_val in col_ext_inputs.get(layer_name, {}).items():
                    final_inputs[layer_name].setdefault(in_type, 0)
                    final_inputs[layer_name][in_type] += in_val

            col.update(final_inputs)

        # --- 3. Actualizar el estado de la red ---
        for i, col in enumerate(self.columns):
            self.network_state[f'col_{i}'] = col.get_state()

    def record_weights(self):
        """Le pide al ConnectionManager que guarde los pesos actuales."""
        self.connection_manager.record_weights()

    def apply_plasticity(self, plasticity_rule_func):
        """
        Aplica una regla de plasticidad a todas las conexiones de la red.

        Args:
            plasticity_rule_func (function): Una función que toma (manager, pre_state, post_state, rule)
                                             y devuelve el nuevo peso.
        """
        # Necesitamos el estado pre-sináptico (el actual) y el post-sináptico (el siguiente).
        # Para simplificar, basaremos la plasticidad solo en el estado actual.

        for rule in self.connection_manager.coupling_rules:
            # Identificar neuronas pre y post sinápticas
            # (Simplificación: usamos la actividad promedio de la capa)
            post_col_idx = rule['target_col']
            post_layer_name = rule['target_layer']

            # El estado post-sináptico es la actividad de la capa destino
            post_activity = self.network_state[f'col_{post_col_idx}'][post_layer_name]

            # El estado pre-sináptico es la actividad de las capas fuente
            pre_activities = [self.network_state[f'col_{c}'][l] for c, l in rule['sources']]

            # Obtener el peso actual
            source_col, source_layer = rule['sources'][0] # Simplificación para la firma
            current_weight = self.connection_manager.get_weight(source_col, source_layer, post_col_idx, post_layer_name)

            # Calcular el nuevo peso usando la regla
            new_weight = plasticity_rule_func(current_weight, pre_activities, post_activity)

            # Actualizar el peso en el ConnectionManager
            self.connection_manager.update_weight(source_col, source_layer, post_col_idx, post_layer_name, new_weight)

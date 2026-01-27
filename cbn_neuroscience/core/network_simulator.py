# cbn_neuroscience/core/network_simulator.py

import numpy as np
from cbn_neuroscience.core.connections import ConnectionManager
from cbn_neuroscience.core.plasticity_manager import PlasticityManager

class NetworkSimulator:
    def __init__(self, columns, coupling_rules, plasticity_manager: PlasticityManager = None):
        self.columns = columns
        self.connection_manager = ConnectionManager(columns, coupling_rules)
        self.dt = columns[0].layers[list(columns[0].layers.keys())[0]].dt if columns else 0.1
        self.plasticity_manager = plasticity_manager

        # Inicialización para la regla de covarianza
        if self.plasticity_manager and self.plasticity_manager.rule_type == 'covariance':
            self.avg_rates = {} # (col_idx, layer_name) -> avg_rate
            self.tau_avg_rate = 100.0 # ms, constante de tiempo para el promedio móvil

    def run_step(self, step_idx, ext_inputs: dict):
        step_time = step_idx * self.dt

        # Obtener el estado de spikes del paso anterior
        prev_spikes_state = {f'col_{i}': col.get_state() for i, col in enumerate(self.columns)}

        # Calcular inputs para cada columna
        for i, col in enumerate(self.columns):
            layer_inputs = {name: {'exc_spikes': np.zeros(layer.n_nodes), 'inh_spikes': np.zeros(layer.n_nodes), 'I_noise': np.zeros(layer.n_nodes)}
                            for name, layer in col.layers.items()}

            # Procesar reglas
            for rule in self.connection_manager.coupling_rules:
                if rule['target_col'] != i: continue

                target_layer_name = rule['target_layer']
                rule_type = rule.get('type', 'additive')

                if rule_type == 'additive':
                    for source_col, source_layer in rule['sources']:
                        source_activity = np.mean(prev_spikes_state[f'col_{source_col}'][source_layer])
                        if source_activity == 0: continue

                        weight = self.connection_manager.get_weight(source_col, source_layer, i, target_layer_name)

                        if col.is_spike_based:
                            if weight >= 0:
                                layer_inputs[target_layer_name]['exc_spikes'] += weight * source_activity
                            else:
                                layer_inputs[target_layer_name]['inh_spikes'] += -weight * source_activity
                        else: # Rate-based
                            layer_inputs[target_layer_name].setdefault('I_total', 0)
                            layer_inputs[target_layer_name]['I_total'] += weight * source_activity

                elif rule_type == 'multiplicative':
                    product_of_sources = 1.0
                    for source_col, source_layer in rule['sources']:
                        product_of_sources *= np.mean(prev_spikes_state[f'col_{source_col}'][source_layer])

                    if product_of_sources == 0: continue

                    # Para la lógica multiplicativa, asumimos que el peso es el mismo desde todas las fuentes
                    source_col, source_layer = rule['sources'][0]
                    weight = self.connection_manager.get_weight(source_col, source_layer, i, target_layer_name)

                    if not col.is_spike_based:
                        layer_inputs[target_layer_name].setdefault('I_total', 0)
                        layer_inputs[target_layer_name]['I_total'] += weight * product_of_sources

            # Añadir inputs externos y actualizar la columna
            col_ext_inputs = ext_inputs.get(i, {})
            for layer_name, inputs in col_ext_inputs.items():
                if col.is_spike_based:
                    for input_type, val in inputs.items():
                        layer_inputs[layer_name][input_type] += val
                else: # Rate-based, sumar todo a I_total
                    layer_inputs[layer_name].setdefault('I_total', 0)
                    layer_inputs[layer_name]['I_total'] += inputs.get('I_noise', 0) # El test usa I_noise

            # Unificar la llamada de actualización
            for layer_name, layer in col.layers.items():
                inputs_for_layer = layer_inputs.get(layer_name, {})
                if col.is_spike_based:
                    layer.update(step_time, **inputs_for_layer)
                else:
                    layer.update(**inputs_for_layer)

        # Actualizar tasas promedio para reglas de covarianza
        if self.plasticity_manager and self.plasticity_manager.rule_type == 'covariance':
            alpha = self.dt / self.tau_avg_rate
            for i, col in enumerate(self.columns):
                for layer_name, layer in col.layers.items():
                    key = (i, layer_name)
                    current_rate = np.mean(layer.A)

                    if key not in self.avg_rates:
                        self.avg_rates[key] = current_rate
                    else:
                        self.avg_rates[key] += alpha * (current_rate - self.avg_rates[key])

        # Aplicar plasticidad
        if self.plasticity_manager:
            self.apply_plasticity(step_time)

    def apply_plasticity(self, step_time):
        """Aplica la regla de plasticidad configurada."""
        if self.plasticity_manager.rule_type == 'stdp_multiplicative':
            self._apply_stdp_multiplicative(step_time)
        elif self.plasticity_manager.rule_type == 'covariance':
            self._apply_covariance_rule()

    def _apply_covariance_rule(self):
        """Aplica la regla de plasticidad de covarianza."""
        for rule in self.connection_manager.coupling_rules:
            # Esta regla es para modelos de tasa
            post_col_idx, post_layer_name = rule['target_col'], rule['target_layer']
            post_layer = self.columns[post_col_idx].layers[post_layer_name]

            # La regla de covarianza puede tener múltiples fuentes
            for source_col_idx, source_layer_name in rule['sources']:
                pre_layer = self.columns[source_col_idx].layers[source_layer_name]

                pre_rate = np.mean(pre_layer.A)
                post_rate = np.mean(post_layer.A)
                pre_avg_rate = self.avg_rates.get((source_col_idx, source_layer_name), 0)
                post_avg_rate = self.avg_rates.get((post_col_idx, post_layer_name), 0)

                dw = self.plasticity_manager.calculate_dw(
                    pre_rate=pre_rate, post_rate=post_rate,
                    pre_avg_rate=pre_avg_rate, post_avg_rate=post_avg_rate
                )

                if dw != 0:
                    current_weight = self.connection_manager.get_weight(source_col_idx, source_layer_name, post_col_idx, post_layer_name)
                    new_weight = current_weight + dw

                    w_max = self.plasticity_manager.params.get('w_max', 1.0)
                    w_min = self.plasticity_manager.params.get('w_min', 0.0)
                    new_weight = np.clip(new_weight, w_min, w_max)

                    self.connection_manager.update_weight(source_col_idx, source_layer_name, post_col_idx, post_layer_name, new_weight)

    def _apply_stdp_multiplicative(self, step_time):
        """Aplica STDP con cruce de dominio."""
        for rule in self.connection_manager.coupling_rules:
            post_col_idx, post_layer_name = rule['target_col'], rule['target_layer']
            post_layer = self.columns[post_col_idx].layers[post_layer_name]

            source_col_idx, source_layer_name = rule['sources'][0]
            pre_layer = self.columns[source_col_idx].layers[source_layer_name]

            # Solo aplicar si hay spikes en alguna de las dos capas para eficiencia
            if not (np.any(pre_layer.spikes) or np.any(post_layer.spikes)):
                continue

            # Comparar cada neurona pre con cada neurona post (simplificación)
            # Un modelo más detallado tendría una matriz de pesos por neurona.
            # Aquí, el cambio de peso se promedia.

            total_dw = 0.0
            interaction_count = 0

            for post_idx in np.where(post_layer.spikes)[0]:
                t_post = post_layer.last_spike_time[post_idx]

                # Para un t_post, buscar el t_pre más cercano
                all_t_pre = pre_layer.last_spike_time
                if np.all(all_t_pre < 0): continue # No hay spikes pre

                # Encontrar el pre-spike más cercano en el tiempo
                time_diffs = t_post - all_t_pre
                closest_pre_idx = np.argmin(np.abs(time_diffs))
                delta_t = time_diffs[closest_pre_idx]

                if -40 < delta_t < 40 and delta_t != 0:
                    current_weight = self.connection_manager.get_weight(source_col_idx, source_layer_name, post_col_idx, post_layer_name)
                    dw = self.plasticity_manager.calculate_dw(w=current_weight, delta_t=delta_t)
                    total_dw += dw
                    interaction_count += 1

            if interaction_count > 0:
                avg_dw = total_dw / interaction_count
                current_weight = self.connection_manager.get_weight(source_col_idx, source_layer_name, post_col_idx, post_layer_name)
                new_weight = current_weight + avg_dw

                w_max = self.plasticity_manager.params.get('w_max', 1.0)
                w_min = self.plasticity_manager.params.get('w_min', -1.0)
                new_weight = np.clip(new_weight, w_min, w_max)

                self.connection_manager.update_weight(source_col_idx, source_layer_name, post_col_idx, post_layer_name, new_weight)

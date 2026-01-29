# cbn_neuroscience/core/connections.py

import numpy as np

class ConnectionManager:
    """
    Gestiona la conectividad y los pesos dinámicos de la red.
    """
    def __init__(self, columns, coupling_rules):
        self.columns = columns
        self.coupling_rules = coupling_rules

        # Crear un mapa de 'capa_global' a índice para la matriz de pesos
        self.layer_map = {}
        idx = 0
        for i, col in enumerate(self.columns):
            for layer_name in col.layers.keys():
                self.layer_map[(i, layer_name)] = idx
                idx += 1

        num_layers = len(self.layer_map)
        self.weights = np.zeros((num_layers, num_layers))

        # Inicializar los pesos según las reglas
        self._initialize_weights()

        # Monitor de evolución de pesos
        self.weight_history = []

    def _initialize_weights(self):
        """
        Puebla la matriz de pesos inicial a partir de la lista de reglas.
        """
        for rule in self.coupling_rules:
            target_idx = self.layer_map.get((rule['target_col'], rule['target_layer']))
            if target_idx is None: continue

            # Para simplificar, las reglas con múltiples fuentes apuntarán
            # a la misma entrada de la matriz de pesos.
            # Una implementación más compleja podría tener pesos por cada fuente.
            for source_col, source_layer in rule['sources']:
                source_idx = self.layer_map.get((source_col, source_layer))
                if source_idx is None: continue

                # Asignar el peso inicial
                self.weights[target_idx, source_idx] = rule.get('weight', 1.0)

    def get_weight(self, source_col, source_layer, target_col, target_layer):
        """Obtiene un peso específico de la matriz."""
        source_idx = self.layer_map[(source_col, source_layer)]
        target_idx = self.layer_map[(target_col, target_layer)]
        return self.weights[target_idx, source_idx]

    def update_weight(self, source_col, source_layer, target_col, target_layer, new_weight):
        """Actualiza un peso específico."""
        source_idx = self.layer_map[(source_col, source_layer)]
        target_idx = self.layer_map[(target_col, target_layer)]
        self.weights[target_idx, source_idx] = new_weight

    def record_weights(self):
        """Guarda una copia de la matriz de pesos actual en el historial."""
        self.weight_history.append(self.weights.copy())

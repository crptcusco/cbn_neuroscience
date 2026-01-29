# cbn_neuroscience/core/plasticity_manager.py

import numpy as np

class PlasticityManager:
    """
    Gestiona diferentes reglas de aprendizaje para la plasticidad sináptica.
    """
    def __init__(self, rule_type='stdp_multiplicative', **params):
        self.rule_type = rule_type
        self.params = params

    def calculate_dw(self, **kwargs):
        """Despachador para la regla de aprendizaje correcta."""
        if self.rule_type == 'stdp_multiplicative':
            return self._calculate_dw_stdp_multiplicative(**kwargs)
        elif self.rule_type == 'covariance':
            return self._calculate_dw_covariance(**kwargs)
        return 0.0

    def _calculate_dw_stdp_multiplicative(self, w, delta_t):
        """
        Regla STDP multiplicativa con límites suaves y cruce de dominio.
        """
        a_plus = self.params.get('a_plus', 0.1)
        a_minus = self.params.get('a_minus', -0.1) # a_minus debe ser negativo
        tau_plus = self.params.get('tau_plus', 20.0)
        tau_minus = self.params.get('tau_minus', 20.0)
        w_max = self.params.get('w_max', 1.0)
        w_min = self.params.get('w_min', -1.0)

        if delta_t > 0:
            # LTP: el peso se acerca a w_max
            dw = a_plus * (w_max - w) * np.exp(-delta_t / tau_plus)
        elif delta_t < 0:
            # LTD: el peso se acerca a w_min (puede ser negativo)
            dw = a_minus * (w - w_min) * np.exp(delta_t / tau_minus)
        else:
            dw = 0.0
        return dw

    def _calculate_dw_covariance(self, pre_rate, post_rate, pre_avg_rate, post_avg_rate):
        """
        Regla de covarianza para poblaciones (Eq. 6.11).
        """
        learning_rate = self.params.get('learning_rate', 0.01)

        # dw = eta * (r_i - <r_i>) * (r_j - <r_j>)
        dw = learning_rate * (pre_rate - pre_avg_rate) * (post_rate - post_avg_rate)
        return dw

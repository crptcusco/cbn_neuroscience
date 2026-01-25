# cbn_neuroscience/core/lif_nodegroup.py
import numpy as np

class LIF_NodeGroup:
    """
    Modelo GLIF (Generalized Leaky Integrate-and-Fire) con conductancias sinápticas.
    Los spikes entrantes modulan una conductancia que decae exponencialmente,
    creando una corriente sináptica persistente (Eq. 5.14, Trappenberg).
    """
    def __init__(self, n_nodes, tau_m=15.0, theta=-55.0, v_reset=-70.0, v_rest=-70.0,
                 R_m=10.0, tau_syn_exc=5.0, E_exc=0.0, delta=2.0, dt=0.1):
        self.n_nodes = n_nodes
        self.tau_m = tau_m
        self.theta = theta
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.R_m = R_m
        self.E_exc = E_exc  # Potencial de reversión excitatorio
        self.delta_steps = int(delta / dt)
        self.dt = dt

        self.v = np.full(n_nodes, self.v_rest)
        self.spikes = np.zeros(n_nodes, dtype=bool)
        self.refractory_timer = np.zeros(n_nodes, dtype=int)

        # Estado de la conductancia sináptica
        self.g_exc = np.zeros(n_nodes)
        self.syn_decay_exc = np.exp(-dt / tau_syn_exc)

    def update(self, weighted_spikes, I_noise):
        """
        Args:
            weighted_spikes (np.ndarray): Suma de los pesos de los spikes de entrada.
                                          Esto incrementa la conductancia.
            I_noise (np.ndarray): Ruido de corriente.
        """
        # 1. Actualizar la conductancia sináptica (decae y se incrementa por spikes)
        self.g_exc *= self.syn_decay_exc
        self.g_exc += weighted_spikes

        # 2. Calcular la corriente sináptica efectiva.
        # La conductancia g_exc se trata como la corriente de entrada.
        I_syn = self.g_exc

        # 3. Actualizar el potencial de membrana (ecuación LIF con R_m)
        in_refractory = self.refractory_timer > 0

        dv = (-(self.v[~in_refractory] - self.v_rest) + self.R_m * I_syn[~in_refractory]) / self.tau_m
        self.v[~in_refractory] += dv * self.dt + I_noise[~in_refractory]

        self.refractory_timer[in_refractory] -= 1

        # 4. Detectar y resetear spikes
        self.spikes = self.v >= self.theta
        if np.any(self.spikes):
            self.v[self.spikes] = self.v_reset
            self.refractory_timer[self.spikes] = self.delta_steps

# cbn_neuroscience/core/lif_nodegroup.py
import numpy as np
from cbn_neuroscience.core.neuron_model import NeuronModel

class LIF_NodeGroup(NeuronModel):
    """
    Modelo LIF con corriente sináptica de decaimiento exponencial.
    Los spikes entrantes generan una corriente I_syn que decae en el tiempo.
    """
    def __init__(self, n_nodes, tau_m=15.0, theta=-55.0, v_reset=-70.0, v_rest=-70.0,
                 R_m=10.0, tau_syn_exc=5.0, delta=2.0, dt=0.1, **kwargs):
        super().__init__(n_nodes)
        self.tau_m = tau_m
        self.theta = theta
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.R_m = R_m
        self.delta_steps = int(delta / dt)
        self.dt = dt

        self.v = np.full(n_nodes, self.v_rest)
        self.spikes = np.zeros(n_nodes, dtype=bool)
        self.refractory_timer = np.zeros(n_nodes, dtype=int)

        # Corriente sináptica (estado interno)
        self.I_syn_exc = np.zeros(n_nodes)
        self.I_syn_nmda = np.zeros(n_nodes)
        self.syn_decay_exc = np.exp(-dt / tau_syn_exc)

    def update(self, **inputs):
        """
        Args (via dict):
            weighted_spikes (np.ndarray): Spikes de entrada AMPA (rápidos).
            nmda_spikes (np.ndarray): Spikes de entrada NMDA (dependientes de voltaje).
            I_noise (np.ndarray): Ruido de corriente.
        """
        weighted_spikes = np.full(self.n_nodes, inputs.get('weighted_spikes', 0))
        nmda_spikes = np.full(self.n_nodes, inputs.get('nmda_spikes', 0))
        I_noise = np.full(self.n_nodes, inputs.get('I_noise', 0))
        # 1. Actualizar corrientes sinápticas
        self.I_syn_exc *= self.syn_decay_exc
        self.I_syn_exc += weighted_spikes

        # 2. Compuerta NMDA: solo integra los spikes si la neurona está despolarizada
        is_depolarized = self.v > -60.0
        self.I_syn_nmda *= self.syn_decay_exc # Usar el mismo decaimiento por simplicidad
        self.I_syn_nmda[is_depolarized] += nmda_spikes[is_depolarized]

        # 3. Corriente total y actualización del potencial de membrana
        total_synaptic_current = self.I_syn_exc + self.I_syn_nmda
        in_refractory = self.refractory_timer > 0

        dv = (-(self.v[~in_refractory] - self.v_rest) + self.R_m * total_synaptic_current[~in_refractory]) / self.tau_m
        self.v[~in_refractory] += dv * self.dt + I_noise[~in_refractory]

        self.refractory_timer[in_refractory] -= 1

        # 4. Detectar y resetear spikes
        self.spikes = self.v >= self.theta
        if np.any(self.spikes):
            self.v[self.spikes] = self.v_reset
            self.refractory_timer[self.spikes] = self.delta_steps

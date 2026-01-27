# cbn_neuroscience/core/lif_nodegroup.py
import numpy as np
from cbn_neuroscience.core.neuron_model import NeuronModel

class LIF_NodeGroup(NeuronModel):
    """
    Modelo LIF con sinapsis excitatorias e inhibitorias separadas.
    """
    def __init__(self, n_nodes, tau_m=15.0, theta=-55.0, v_reset=-70.0, v_rest=-70.0,
                 R_m=10.0, tau_syn_exc=5.0, tau_syn_inh=10.0, delta=2.0, dt=0.1, **kwargs):
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
        self.last_spike_time = np.full(n_nodes, -np.inf)

        self.I_syn_exc = np.zeros(n_nodes)
        self.I_syn_inh = np.zeros(n_nodes)
        self.syn_decay_exc = np.exp(-dt / tau_syn_exc)
        self.syn_decay_inh = np.exp(-dt / tau_syn_inh)

    def update(self, step_time, **inputs):
        """
        Args (via dict):
            exc_spikes (np.ndarray): Spikes de entrada excitatorios.
            inh_spikes (np.ndarray): Spikes de entrada inhibitorios.
            I_noise (np.ndarray): Ruido de corriente.
        """
        exc_spikes = np.full(self.n_nodes, inputs.get('exc_spikes', 0))
        inh_spikes = np.full(self.n_nodes, inputs.get('inh_spikes', 0))
        I_noise = np.full(self.n_nodes, inputs.get('I_noise', 0))

        # 1. Actualizar corrientes sinápticas
        self.I_syn_exc = self.I_syn_exc * self.syn_decay_exc + exc_spikes
        self.I_syn_inh = self.I_syn_inh * self.syn_decay_inh + inh_spikes

        # 2. Corriente total y actualización del potencial de membrana
        total_synaptic_current = self.I_syn_exc - self.I_syn_inh # La inhibición es sustractiva
        in_refractory = self.refractory_timer > 0

        dv = (-(self.v[~in_refractory] - self.v_rest) + self.R_m * total_synaptic_current[~in_refractory]) / self.tau_m
        self.v[~in_refractory] += dv * self.dt + I_noise[~in_refractory]

        self.refractory_timer[in_refractory] -= 1

        # --- Detección y registro de spikes ---
        self.spikes = self.v >= self.theta
        if np.any(self.spikes):
            self.v[self.spikes] = self.v_reset
            self.refractory_timer[self.spikes] = self.delta_steps
            self.last_spike_time[self.spikes] = step_time

# cbn_neuroscience/core/srm_nodegroup.py
import numpy as np

class SRM_NodeGroup:
    """
    Representa un grupo de neuronas usando el Spike-Response Model (SRM).
    La dinámica se basa en la suma de kernels, como se describe en el
    Capítulo 5.1.3 de Trappenberg (Eqs. 5.12 - 5.17).
    """
    def __init__(self, n_nodes, tau_m=15.0, theta=-55.0, v_reset=-70.0, v_rest=-70.0, dt=0.1):
        """
        Inicializa el grupo de neuronas SRM.

        Args:
            n_nodes (int): Número de neuronas en el grupo.
            tau_m (float): Constante de tiempo de membrana (ms).
            theta (float): Umbral de disparo (mV).
            v_reset (float): Potencial de reseteo post-disparo (mV).
            v_rest (float): Potencial de membrana en reposo (mV).
            dt (float): Paso de tiempo de la simulación (ms).
        """
        self.n_nodes = n_nodes
        self.tau_m = tau_m
        self.theta = theta
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt

        # Factores de decaimiento pre-calculados para eficiencia
        self.syn_decay = np.exp(-dt / tau_m)
        self.ref_decay = np.exp(-dt / tau_m)

        # Estado de las neuronas
        self.v = np.full(n_nodes, self.v_rest)
        self.spikes = np.zeros(n_nodes, dtype=bool)

        # Estados internos para los kernels (potenciales sináptico y refractario)
        self.h_syn = np.zeros(n_nodes)
        self.h_ref = np.zeros(n_nodes)

    def update(self, weighted_input_spikes, noise_term):
        """
        Actualiza el estado del grupo de neuronas para un paso de tiempo.

        Args:
            weighted_input_spikes (np.ndarray): Suma de los spikes de entrada pesados.
            noise_term (np.ndarray): Término de ruido gaussiano ξ(t).
        """
        # 1. Aplicar decaimiento exponencial a los potenciales existentes
        self.h_syn *= self.syn_decay
        self.h_ref *= self.ref_decay

        # 2. Integrar nuevos spikes de entrada (kernel épsilon)
        self.h_syn += weighted_input_spikes

        # 3. Calcular el potencial de membrana total con ruido
        self.v = self.v_rest + self.h_syn + self.h_ref + noise_term

        # 4. Detectar disparos (spikes)
        self.spikes = self.v >= self.theta

        # 5. Aplicar el kernel de reseteo (eta) para las neuronas que dispararon
        if np.any(self.spikes):
            # El reseteo anula el potencial sináptico y fija el voltaje a v_reset.
            # h_ref = v_reset - v_rest - h_syn
            self.h_ref[self.spikes] = self.v_reset - self.v_rest - self.h_syn[self.spikes]

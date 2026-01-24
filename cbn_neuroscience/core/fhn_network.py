# cbn_neuroscience/core/fhn_network.py

import numpy as np

class FHN_NodeGroup:
    """
    Un grupo de nodos FitzHugh-Nagumo vectorizados.
    """
    def __init__(self, n_nodes, a=0.7, b=0.8, dt=0.1):
        self.n_nodes = n_nodes
        self.a = np.full(n_nodes, a)
        self.b = np.full(n_nodes, b)
        self.dt = dt

        self.v = np.random.uniform(-1.2, -1.0, n_nodes)
        self.w = np.random.uniform(-0.6, -0.4, n_nodes)
        self.states = np.zeros(n_nodes, dtype=int)

    def update(self, I_ext):
        """
        Actualiza el estado de todos los nodos en el grupo.
        """
        dv = self.v - (self.v**3 / 3) - self.w + I_ext
        dw = 0.08 * (self.v + self.a - self.b * self.w)

        self.v += self.dt * dv
        self.w += self.dt * dw

        # El estado booleano se activa si el potencial de membrana cruza un umbral
        self.states = (self.v > 1.0).astype(int)

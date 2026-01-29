# cbn_neuroscience/core/neuron_model.py
from abc import ABC, abstractmethod

class NeuronModel(ABC):
    """
    Clase base abstracta para todos los modelos neuronales.
    Define la interfaz que CompartmentalColumn utilizará.
    """
    @abstractmethod
    def __init__(self, n_nodes, **kwargs):
        self.n_nodes = n_nodes
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Actualiza el estado interno del modelo neuronal para un paso de tiempo.
        """
        pass

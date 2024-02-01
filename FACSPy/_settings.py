
from typing import Union

class FACSPyConfig:
    """
    Config Manager, inspired by scanpy
    """

    def __init__(self,
                 *,
                 n_jobs = 1,
                 layer = "compensated",
                 gate = ""
                 ):
        self.n_jobs = n_jobs
        self.default_layer = layer
        self.default_gate = gate

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self,
               n_jobs: Union[int, float]):
        self._n_jobs = n_jobs

    @property
    def default_layer(self):
        return self._default_layer
    
    @default_layer.setter
    def default_layer(self,
                      layer):
        self._default_layer = layer

    @property
    def default_gate(self):
        return self._default_gate

    @default_gate.setter
    def default_gate(self,
                     gate: str):
        self._default_gate = gate

    @property
    def tight_layout(self):
        return self._tight_layout

    @default_gate.setter
    def tight_layout(self,
                     value : bool):
        self._default_gate = value


settings = FACSPyConfig()
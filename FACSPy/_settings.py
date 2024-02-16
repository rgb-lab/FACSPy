
from typing import Union

class FACSPyConfig:
    """
    Config Manager, inspired by scanpy
    """

    def __init__(self,
                 *,
                 n_jobs = 1,
                 layer = "compensated",
                 gate = "",
                 tight_layout = False,
                 default_categorical_cmap = "Set1",
                 gate_aliases = None
                 ):
        self.n_jobs = n_jobs
        self.default_layer = layer
        self.default_gate = gate
        self.tight_layout = tight_layout
        self.default_categorical_cmap = default_categorical_cmap
        if gate_aliases is None:
            self.gate_aliases = dict()
        else:
            self.gate_aliases = gate_aliases

    @property
    def gate_aliases(self):
        return self._gate_aliases
    
    @gate_aliases.setter
    def gate_aliases(self,
                     gate_aliases):
        self._gate_aliases = gate_aliases

    def add_new_alias(self,
                      gate_name,
                      gate_alias) -> None:
        self.gate_aliases[gate_alias] = gate_name
    
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

    @tight_layout.setter
    def tight_layout(self,
                     value : bool):
        self._tight_layout = value

    @property
    def default_categorical_cmap(self):
        return self._default_categorical_cmap

    @default_categorical_cmap.setter
    def default_categorical_cmap(self,
                                 cmap: str):
        self._default_categorical_cmap = cmap


settings = FACSPyConfig()
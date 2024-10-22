""" Spectral Flow / Flow / Mass Cytometry analysis """

from . import tools as tl
from . import plotting as pl
from . import model as ml
from . import dataset as dt
from . import synchronization as sync


from .dataset import create_dataset, transform
from ._utils import (subset_gate,
                     subset_fluo_channels,
                     remove_unnamed_channels,
                     remove_channel,
                     equalize_groups,
                     convert_gate_to_obs,
                     rename_channel,
                     convert_cluster_to_gate,
                     r_setup,
                     r_restore)
from .io._io import save_dataset, read_dataset
from ._settings import settings, FACSPyConfig

from .datasets._datasets import mouse_lineages


import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pl', 'ml', 'dt', 'sync']})

all = [
    "tl",
    "pl",
    "ml",
    "dt",
    "sync",
    "subset_gate",
    "subset_fluo_channels",
    "remove_unnamed_channels",
    "remove_channel",
    "equalize_groups",
    "convert_gate_to_obs",
    "rename_channel",
    "convert_cluster_to_gate",
    "FACSPyConfig",
    "mouse_lineages"
]

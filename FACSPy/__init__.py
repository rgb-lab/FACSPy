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
                     convert_gates_to_obs,
                     add_metadata_to_obs,
                     rename_channel,
                     convert_cluster_to_gate,
                     is_fluo_channel)
from .io._io import save_dataset, read_dataset
from ._settings import settings

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pl', 'ml', 'dt']})
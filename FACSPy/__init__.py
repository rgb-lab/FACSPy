""" Spectral Flow / Flow / Mass Cytometry analysis """

from . import tools as tl
from . import plotting as pl
from . import model as ml
from . import dataset as dt

from .utils import subset_gate, subset_fluo_channels, remove_unnamed_channels, equalize_groups
from .io.io import save_dataset, read_dataset

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pl', 'ml', 'dt']})
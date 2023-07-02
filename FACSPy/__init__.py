""" Spectral Flow / Flow / Mass Cytometry analysis """

from . import tools as tl
from . import plotting as pl
from . import model as ml

import sys

sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pl', 'ml']})
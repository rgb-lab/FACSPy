import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Literal, Union, Optional

from .utils import (prep_uns_dataframe,
                    turn_off_missing_plots,
                    scale_data,
                    select_gate_from_singleindex_dataframe,
                    map_obs_to_cmap,
                    calculate_metaclusters,
                    map_metaclusters_to_sample_ID,
                    merge_metaclusters_into_dataframe,
                    ANNOTATION_CMAPS)
from ..utils import find_gate_path_of_gate, reduction_names, subset_gate

from ..exceptions.exceptions import AnalysisNotPerformedError

from scipy.spatial import distance
from scipy.cluster import hierarchy


def create_clustermap(data: Union[pd.DataFrame, np.ndarray],
                      row_colors: Optional[list[Union[str, int, pd.Series]]] = None,
                      col_colors: Optional[list[Union[str, int, pd.Series]]] = None,
                      row_linkage: Optional[np.ndarray] = None,
                      col_linkage: Optional[np.ndarray] = None,
                      figsize: Optional[tuple[float, float]] = (4,4),
                      cmap: Optional[str] = "inferno",
                      vmin: Optional[int] = None,
                      vmax: Optional[int] = None,
                      cbar_kws: Optional[dict] = None) -> sns.matrix.ClusterGrid:
    return sns.clustermap(
        data = data,
        row_colors = row_colors,
        col_colors = col_colors,
        row_linkage = row_linkage,
        col_linkage = row_linkage,
        cmap = cmap,
        dendrogram_ratio = (0.1, 0.1),
        annot_kws = {"size": 4},
        figsize = figsize,
        yticklabels = True,
        xticklabels = True,
        vmin = vmin,
        vmax = vmax,
        cbar_kws = cbar_kws
    )
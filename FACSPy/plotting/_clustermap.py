import pandas as pd
import numpy as np
import seaborn as sns
from typing import Union, Optional

def create_clustermap(data: Union[pd.DataFrame, np.ndarray],
                      row_colors: Optional[list[Union[str, int, pd.Series]]] = None,
                      col_colors: Optional[list[Union[str, int, pd.Series]]] = None,
                      row_linkage: Optional[np.ndarray] = None,
                      col_linkage: Optional[np.ndarray] = None,
                      row_cluster: Optional[bool] = True,
                      col_cluster: Optional[bool] = True,
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
        col_linkage = col_linkage,
        row_cluster = row_cluster,
        col_cluster = col_cluster,
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
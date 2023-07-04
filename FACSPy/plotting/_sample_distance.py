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
                    calculate_sample_distance,
                    calculate_linkage,
                    add_metaclusters,
                    remove_ticklabels,
                    remove_ticks,
                    scale_cbar_to_heatmap,
                    add_categorical_legend_to_clustermap,
                    ANNOTATION_CMAPS)

from ..utils import find_gate_path_of_gate

from ..exceptions.exceptions import AnalysisNotPerformedError

from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix

from ._clustermap import create_clustermap


def sample_distance(adata: AnnData,
                    groupby: Optional[Union[str, list[str]]],
                    gate: str,
                    scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                    on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    cmap: str = "inferno",
                    figsize: tuple[float, float] = (4,4),
                    return_fig: bool = False,
                    return_dataframe: bool = False,
                    metaclusters: Optional[int] = None,
                    label_metaclusters_in_dataset: bool = True,
                    label_metaclusters_key: Optional[str] = "metacluster_sc") -> Optional[Figure]:
    
    try:
        data = adata.uns[on]
        data = prep_uns_dataframe(adata, data)
        data = select_gate_from_singleindex_dataframe(data, find_gate_path_of_gate(adata, gate))
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]
        if scaling is not None:
            data[fluo_columns] = scale_data(data[fluo_columns], scaling)

    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e
    
    if return_dataframe:
        return data
    
    if not isinstance(groupby, list):
        groupby = [groupby]

    sample_IDs = data["sample_ID"].to_list()
    distance_data = calculate_sample_distance(data[fluo_columns])
    row_linkage = calculate_linkage(distance_data)

    if metaclusters is not None:
        groupby += ["metacluster"]
        data = add_metaclusters(adata = adata,
                                row_linkage = row_linkage,
                                n_clusters = metaclusters,
                                sample_IDs = sample_IDs,
                                label_metaclusters = label_metaclusters_in_dataset,
                                label_metaclusters_key = label_metaclusters_key)
    
    clustermap = create_clustermap(data = distance_data,
                                   row_colors = [
                                       map_obs_to_cmap(data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(groupby)
                                   ],
                                   col_colors = [
                                       map_obs_to_cmap(data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(groupby)
                                   ],
                                   row_linkage = row_linkage,
                                   col_linkage = row_linkage,
                                   cmap = cmap,
                                    figsize = figsize
                                   )
    
    ax = clustermap.ax_heatmap
    heatmap_position = ax.get_position()
    
    scale_cbar_to_heatmap(clustermap,
                          heatmap_position = heatmap_position)
    remove_ticklabels(ax, which = "both")
    remove_ticks(ax, which = "both")
    add_categorical_legend_to_clustermap(clustermap,
                                         heatmap = ax,
                                         data = data,
                                         groupby = groupby)
    
    if return_fig:
        return clustermap
    
    plt.show()
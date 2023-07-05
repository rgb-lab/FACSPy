from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib.patches import Patch

from typing import Literal, Optional, Union

from ..utils import subset_gate, find_gate_path_of_gate
from .utils import (prep_uns_dataframe,
                    scale_data,
                    select_gate_from_singleindex_dataframe,
                    map_obs_to_cmap,
                    calculate_sample_distance,
                    calculate_linkage,
                    add_metaclusters,
                    remove_ticklabels,
                    remove_ticks,
                    scale_cbar_to_heatmap,
                    add_categorical_legend_to_clustermap,
                    calculate_correlation_data,
                    remove_dendrogram,
                    add_annotation_plot,
                    ANNOTATION_CMAPS)

from ._clustermap import create_clustermap

from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix



from ..exceptions.exceptions import AnalysisNotPerformedError

def expression_heatmap(adata: AnnData,
                       groupby: Optional[Union[str, list[str]]],
                       gate: str,
                       annotate: Optional[str] = None,
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cluster_method: Literal["correlation", "distance"] = "distance",
                       cmap: str = "inferno",
                       return_fig: bool = False,
                       metaclusters: Optional[int] = None,
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc",
                       figsize: Optional[tuple[int, int]] = (5,3.8),
                       y_label_fontsize: Optional[Union[int, float]] = 4) -> Optional[Figure]:
    
    try:
        if on == "all_cells":
            adata = subset_gate(adata, gate, as_view = True)
            data = adata.to_df(layer = "transformed")

        else:
            data = adata.uns[on]
            data = prep_uns_dataframe(adata, data)
            data = select_gate_from_singleindex_dataframe(data, find_gate_path_of_gate(adata, gate))
            annot_data = data.copy()
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]
        if scaling is not None:
            data[fluo_columns] = scale_data(data[fluo_columns], scaling)
    
    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e


    if not isinstance(groupby, list):
        groupby = [groupby]

    sample_IDs = data["sample_ID"].to_list()
    sample_annot = data["sample_ID"].astype("object").to_list()
    raw_data = data[fluo_columns].T
    raw_data.columns = sample_annot

    if cluster_method == "correlation":
        correlation_data = raw_data.corr(method = corr_method)
        row_linkage = calculate_linkage(correlation_data.T)
    
    elif cluster_method == "distance":
        distance_data = calculate_sample_distance(data[fluo_columns])
        row_linkage = calculate_linkage(distance_data)

    if metaclusters is not None:
        groupby += ["metacluster"]
        data = add_metaclusters(adata = adata,
                                data = data,
                                row_linkage = row_linkage,
                                n_clusters = metaclusters,
                                sample_IDs = sample_IDs,
                                label_metaclusters = label_metaclusters_in_dataset,
                                label_metaclusters_key = label_metaclusters_key)

    clustermap = create_clustermap(data = raw_data,
                                   col_colors = [
                                       map_obs_to_cmap(data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(groupby)
                                   ],
                                   row_cluster = True,
                                   col_linkage = row_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   cbar_kws = {"label": "scaled expression", "orientation": 'horizontal'},
                                   vmin = 0 if scaling is not None else None,
                                   vmax = 1 if scaling is not None else None
                                   )

    indices = [t.get_text() for t in np.array(clustermap.ax_heatmap.get_xticklabels())]
    ax = clustermap.ax_heatmap
    scale_cbar_to_heatmap(clustermap,
                          heatmap_position = ax.get_position(),
                          cbar_padding = 0.5)    

    remove_ticklabels(ax, which = "x")
    remove_ticks(ax, which = "x")
    remove_dendrogram(clustermap, which = "y")

    ax.yaxis.set_ticks_position("left")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = y_label_fontsize)
    clustermap.ax_row_dendrogram.set_visible(False)

    add_categorical_legend_to_clustermap(clustermap,
                                         heatmap = ax,
                                         data = data,
                                         groupby = groupby,)

    if annotate is not None:
        if annotate in adata.var_names:
            annot_data["sample_ID"] = annot_data["sample_ID"].astype("object")
            annot_data = annot_data.set_index("sample_ID")
            annot_data.index = [str(idx) for idx in annot_data.index.to_list()]
            annot_frame = annot_data[annotate]
        
        add_annotation_plot(adata = adata,
                            annotate = annotate,
                            annot_frame = annot_frame,
                            indices = indices,
                            clustermap = clustermap,
                            y_label_fontsize = y_label_fontsize,
                            y_label = on)

    if return_fig:
        return clustermap
    plt.show()
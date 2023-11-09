import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Literal, Union, Optional
from ._utils import (scale_data,
                    map_obs_to_cmap,
                    calculate_sample_distance,
                    append_metadata,
                    calculate_linkage,
                    add_metaclusters,
                    remove_ticklabels,
                    remove_ticks,
                    scale_cbar_to_heatmap,
                    add_categorical_legend_to_clustermap,
                    get_uns_dataframe,
                    ANNOTATION_CMAPS,
                    CONTINUOUS_CMAPS,
                    has_interval_index,
                    savefig_or_show)

from ._clustermap import create_clustermap

def prepare_plot_data(adata: AnnData,
                      raw_data: pd.DataFrame,
                      scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]],
                      copy: bool = False
                      ) -> pd.DataFrame:
    plot_data = raw_data.copy() if copy else raw_data
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    if scaling is not None:
        plot_data[fluo_columns] = scale_data(plot_data[fluo_columns], scaling)
    sample_distances = calculate_sample_distance(plot_data[fluo_columns])
    plot_data = pd.DataFrame(data = sample_distances,
                             columns = raw_data["sample_ID"].to_list(),
                             index = raw_data["sample_ID"].to_list())
    plot_data = plot_data.fillna(0)
    plot_data["sample_ID"] = raw_data["sample_ID"].to_list()
    plot_data = append_metadata(adata, plot_data)

    return plot_data

def sample_distance(adata: AnnData,
                    gate: str,
                    annotate: Union[str, list[str]],

                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    data_origin: Literal["compensated", "transformed"] = "transformed",
                    
                    
                    scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                    cmap: str = "inferno",
                    figsize: tuple[float, float] = (4,4),
                    return_dataframe: bool = False,
                    metaclusters: Optional[int] = None,
                    label_metaclusters_in_dataset: bool = True,
                    label_metaclusters_key: Optional[str] = "sample_distance_metaclusters",
                    return_fig: bool = False,
                    save: bool = None,
                    show: bool = None) -> Optional[Figure]:
    
    """ plots sample distance data_metrics as a heatmap """
    
    if not isinstance(annotate, list):
        annotate = [annotate] 
    
    raw_data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = f"{data_metric}_{data_group}_{data_origin}")
    
    plot_data = prepare_plot_data(adata = adata,
                                  raw_data = raw_data,
                                  copy = False,
                                  scaling = scaling)
    
    if return_dataframe:
        return plot_data
    
    row_linkage = calculate_linkage(plot_data[plot_data["sample_ID"].to_list()])

    if metaclusters is not None:
        annotate += ["metacluster"]
        plot_data = add_metaclusters(adata = adata,
                                     data = plot_data,
                                     row_linkage = row_linkage,
                                     n_clusters = metaclusters,
                                     sample_IDs = plot_data["sample_ID"],
                                     label_metaclusters = label_metaclusters_in_dataset,
                                     label_metaclusters_key = label_metaclusters_key)
    
    clustermap = create_clustermap(data = plot_data[plot_data["sample_ID"]],
                                   row_colors = [
                                       map_obs_to_cmap(plot_data,
                                                       group,
                                                       CONTINUOUS_CMAPS[i] if has_interval_index(plot_data[group]) else ANNOTATION_CMAPS[i]
                                                       )
                                       for i, group in enumerate(annotate)
                                   ],
                                   col_colors = [
                                       map_obs_to_cmap(plot_data,
                                                       group,
                                                       CONTINUOUS_CMAPS[i] if has_interval_index(plot_data[group]) else ANNOTATION_CMAPS[i]
                                                                        )
                                       for i, group in enumerate(annotate)
                                   ],
                                   row_linkage = row_linkage,
                                   col_linkage = row_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   cbar_kws = {"label": "distance",
                                               "orientation": 'horizontal'}
                                   )
    
    ax = clustermap.ax_heatmap
    heatmap_position = ax.get_position()
    
    scale_cbar_to_heatmap(clustermap,
                          heatmap_position = heatmap_position)
    remove_ticklabels(ax, which = "both")
    remove_ticks(ax, which = "both")
    add_categorical_legend_to_clustermap(clustermap,
                                         heatmap = ax,
                                         data = plot_data,
                                         annotate = annotate)
    
    if return_fig:
        return clustermap
    savefig_or_show(save = save, show = show)
    plt.show()
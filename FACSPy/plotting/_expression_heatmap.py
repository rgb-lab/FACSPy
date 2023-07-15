from anndata import AnnData
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.figure import Figure

from typing import Literal, Optional, Union

from .utils import (scale_data,
                    map_obs_to_cmap,
                    calculate_sample_distance,
                    calculate_linkage,
                    append_metadata,
                    add_metaclusters,
                    remove_ticklabels,
                    remove_ticks,
                    scale_cbar_to_heatmap,
                    add_categorical_legend_to_clustermap,
                    calculate_correlation_data,
                    remove_dendrogram,
                    add_annotation_plot,
                    get_uns_dataframe,
                    ANNOTATION_CMAPS,
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
    return plot_data

def expression_heatmap(adata: AnnData,
                       gate: str,
                       
                       annotate: Optional[Union[str, list[str]]],

                       data_group: Optional[Union[str, list[str]]] = "sample_ID",
                       data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       data_origin: Literal["compensated", "transformed"] = "transformed",
                       
                       
                       plot_annotate: Optional[str] = None,
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cluster_method: Literal["correlation", "distance"] = "distance",
                       cmap: str = "RdBu_r",
                       metaclusters: Optional[int] = None,
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc",
                       figsize: Optional[tuple[int, int]] = (5,3.8),
                       y_label_fontsize: Optional[Union[int, float]] = 4,
                       return_dataframe: bool = False,
                       return_fig: bool = False,
                       save: bool = None,
                       show: bool = None) -> Optional[Figure]:

    if not isinstance(annotate, list):
        annotate = [annotate]    
    
    raw_data = get_uns_dataframe(adata = adata,
                                 gate = gate,
                                 table_identifier = f"{data_metric}_{data_group}_{data_origin}",
                                 column_identifier_name = "sample_ID")
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    plot_data = prepare_plot_data(adata = adata,
                                  raw_data = raw_data,
                                  copy = True,
                                  scaling = scaling)

    if return_dataframe:
        return plot_data
    
    
    if cluster_method == "correlation":
        col_linkage = calculate_linkage(calculate_correlation_data(plot_data[fluo_columns].T, corr_method))
    
    elif cluster_method == "distance":
        col_linkage = calculate_linkage(calculate_sample_distance(plot_data[fluo_columns]))

    if metaclusters is not None:
        annotate += ["metacluster"]
        plot_data = add_metaclusters(adata = adata,
                                     data = plot_data,
                                     row_linkage = col_linkage,
                                     n_clusters = metaclusters,
                                     sample_IDs = raw_data.index,
                                     label_metaclusters = label_metaclusters_in_dataset,
                                     label_metaclusters_key = label_metaclusters_key)

    ### for the heatmap, the dataframe is transposed so that sample_IDs are the columns
    clustermap = create_clustermap(data = plot_data[fluo_columns].T,
                                   col_colors = [
                                       map_obs_to_cmap(plot_data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(annotate)
                                   ],
                                   row_cluster = True,
                                   col_linkage = col_linkage,
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
    ax.set_xlabel("")

    ax.yaxis.set_ticks_position("left")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = y_label_fontsize)
    clustermap.ax_row_dendrogram.set_visible(False)

    add_categorical_legend_to_clustermap(clustermap,
                                         heatmap = ax,
                                         data = plot_data,
                                         annotate = annotate)

    if plot_annotate is not None:
        if plot_annotate in adata.var_names:
            annot_frame = raw_data[plot_annotate]
        
        add_annotation_plot(adata = adata,
                            annotate = plot_annotate,
                            annot_frame = annot_frame,
                            indices = indices,
                            clustermap = clustermap,
                            y_label_fontsize = y_label_fontsize,
                            y_label = data_metric)

    if return_fig:
        return clustermap
    
    savefig_or_show(save = save, show = show)
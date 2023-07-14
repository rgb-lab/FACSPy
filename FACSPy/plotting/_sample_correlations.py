import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from typing import Literal, Union, Optional

from .utils import (scale_data,
                    map_obs_to_cmap,
                    append_metadata,
                    get_uns_dataframe,
                    calculate_linkage,
                    add_metaclusters,
                    remove_ticklabels,
                    remove_ticks,
                    scale_cbar_to_heatmap,
                    add_categorical_legend_to_clustermap,
                    calculate_correlation_data,
                    ANNOTATION_CMAPS)

from ._clustermap import create_clustermap

def prepare_plot_data(adata: AnnData,
                      raw_data: pd.DataFrame,
                      scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]],
                      corr_method: Literal["pearson", "kendall", "spearman"],
                      copy: bool = False
                      ) -> pd.DataFrame:
    plot_data = raw_data.copy() if copy else raw_data
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    if scaling is not None:
        plot_data[fluo_columns] = scale_data(plot_data[fluo_columns], scaling)
    correlations = calculate_correlation_data(plot_data[fluo_columns].T,
                                              corr_method = corr_method)
    plot_data = pd.DataFrame(data = correlations,
                             columns = raw_data.index.to_list(),
                             index = raw_data.index)
    plot_data = plot_data.fillna(0)
    plot_data = append_metadata(adata, plot_data)

    return plot_data

def sample_correlation(adata: AnnData,
                       groupby: Optional[Union[str, list[str]]],
                       gate: str,
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno",
                       return_fig: bool = False,
                       metaclusters: Optional[int] = None,
                       figsize: tuple[float, float] = (4,4),
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc",
                       return_dataframe: bool = False) -> Optional[Figure]:
    
    if not isinstance(groupby, list):
        groupby = [groupby] 
    
    raw_data = get_uns_dataframe(adata = adata,
                                 gate = gate,
                                 table_identifier = on,
                                 column_identifier_name = "sample_ID")
    
    plot_data = prepare_plot_data(adata = adata,
                                  raw_data = raw_data,
                                  copy = False,
                                  scaling = scaling,
                                  corr_method = corr_method)
    
    if return_dataframe:
        return plot_data

    row_linkage = calculate_linkage(plot_data[plot_data["sample_ID"].to_list()])

    if metaclusters is not None:
        groupby += ["metacluster"]
        plot_data = add_metaclusters(adata = adata,
                                     data = plot_data,
                                     row_linkage = row_linkage,
                                     n_clusters = metaclusters,
                                     sample_IDs = plot_data["sample_ID"],
                                     label_metaclusters = label_metaclusters_in_dataset,
                                     label_metaclusters_key = label_metaclusters_key)
    
    clustermap = create_clustermap(data = plot_data[plot_data["sample_ID"]],
                                   row_colors = [
                                       map_obs_to_cmap(plot_data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(groupby)
                                   ],
                                   col_colors = [
                                       map_obs_to_cmap(plot_data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(groupby)
                                   ],
                                   row_linkage = row_linkage,
                                   col_linkage = row_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   vmin = -1,
                                   vmax = 1,
                                   cbar_kws = {"label": f"{corr_method} correlation",
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
                                         groupby = groupby)
    if return_fig:
        return clustermap
    plt.show()
import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from typing import Literal, Union, Optional

from .utils import (scale_data,
                    map_obs_to_cmap,
                    append_metadata,
                    get_dataframe,
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
    correlations = calculate_correlation_data(plot_data[fluo_columns],
                                              corr_method = corr_method)
    plot_data = pd.DataFrame(data = correlations,
                             columns = fluo_columns,
                             index = fluo_columns)

    return plot_data

def marker_correlation(adata: AnnData,
                       gate: str,
                       scaling: Literal["MinMaxScaler", "RobustScaler"] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency", "all_cells"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno",
                       figsize: tuple[float, float] = (4,4),
                       return_fig: bool = False,
                       return_dataframe: bool = False) -> Optional[Figure]:
    
    raw_data = get_dataframe(adata = adata,
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

    clustermap = create_clustermap(data = plot_data,
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
                          heatmap_position = heatmap_position,
                          cbar_padding = 0.8)
    remove_ticklabels(ax, which = "x")
    remove_ticks(ax, which = "x")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 5)
    if return_fig:
        return clustermap
    plt.show()
import pandas as pd
from anndata import AnnData

from matplotlib.figure import Figure
from typing import Literal, Optional, Union

from ._utils import (_scale_data,
                     _get_uns_dataframe,
                     _remove_ticklabels,
                     _remove_ticks,
                     _scale_cbar_to_heatmap,
                     _calculate_correlation_data,
                     savefig_or_show)

from ._clustermap import create_clustermap

from .._utils import _default_gate_and_default_layer

@_default_gate_and_default_layer
def _prepare_plot_data(adata: AnnData,
                       raw_data: pd.DataFrame,
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]],
                       corr_method: Literal["pearson", "kendall", "spearman"],
                       copy: bool = False
                       ) -> pd.DataFrame:
    plot_data = raw_data.copy() if copy else raw_data
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    if scaling is not None:
        plot_data[fluo_columns] = _scale_data(plot_data[fluo_columns], scaling)
    correlations = _calculate_correlation_data(plot_data[fluo_columns],
                                               corr_method = corr_method)
    correlations = correlations.fillna(0)
    plot_data = pd.DataFrame(data = correlations,
                             columns = fluo_columns,
                             index = fluo_columns)

    return plot_data

def marker_correlation(adata: AnnData,
                       gate: str = None,
                       layer: str = None,
                       scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",

                       data_group: Optional[Union[str, list[str]]] = "sample_ID",
                       data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",

                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno",
                       figsize: tuple[float, float] = (4,4),
                       y_label_fontsize: float = 10,
                       return_fig: bool = False,
                       return_dataframe: bool = False,
                       save: bool = None,
                       show: bool = None) -> Optional[Figure]:
    
    raw_data = _get_uns_dataframe(adata = adata,
                                  gate = gate,
                                  table_identifier = f"{data_metric}_{data_group}_{layer}")
    
    plot_data = _prepare_plot_data(adata = adata,
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
    
    _scale_cbar_to_heatmap(clustermap,
                          heatmap_position = heatmap_position,
                          cbar_padding = 0.8)
    _remove_ticklabels(ax, which = "x")
    _remove_ticks(ax, which = "x")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = y_label_fontsize)
    if return_fig:
        return clustermap

    savefig_or_show(save = save, show = show)
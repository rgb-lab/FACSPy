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

@_default_gate_and_default_layer
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
    """
    Plot for expression heatmap. Rows are the individual channels and columns are the data points.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler`(Z-score)
    corr_method
        correlation method that is used. One of `pearson`, `spearman` or `kendall`.
    y_label_fontsize
        controls the fontsize of the marker labels
    cmap
        Sets the colormap for plotting the markers
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats
    save
        Expects a file path and a file name. saves the figure to the indicated path
    show
        Whether to show the figure
    return_dataframe
        If set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        If set to True, the figure is returned.

    Returns
    -------

    if `show==False` a :class:`~seaborn.ClusterGrid`

    """
 
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
    
    if show is False:
        return clustermap
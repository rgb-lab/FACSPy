import pandas as pd
from anndata import AnnData

import seaborn as sns
from typing import Literal, Optional, Union

from ._utils import (_prepare_heatmap_data,
                     _remove_ticklabels,
                     _remove_ticks,
                     _scale_cbar_to_heatmap,
                     _calculate_correlation_data,
                     savefig_or_show)

from ._clustermap import create_clustermap

from .._utils import _default_gate_and_default_layer, _enable_gate_aliases

def _calculate_correlations(adata: AnnData,
                            plot_data: pd.DataFrame,
                            corr_method: Literal["pearson", "kendall", "spearman"],
                            ) -> pd.DataFrame:
    fluo_columns = [col for col in plot_data.columns if col in adata.var_names]
    correlations = _calculate_correlation_data(plot_data[fluo_columns],
                                               corr_method = corr_method)
    correlations = correlations.fillna(0)
    plot_data = pd.DataFrame(data = correlations,
                             columns = fluo_columns,
                             index = fluo_columns)

    return plot_data

@_default_gate_and_default_layer
@_enable_gate_aliases
def marker_correlation(adata: AnnData,
                       gate: str,
                       layer: str,
                       include_technical_channels: bool = False,
                       exclude: Optional[Union[list[str], str]] = None,
                       scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",
                       data_group: str = "sample_ID",
                       data_metric: Literal["mfi", "fop"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno",
                       y_label_fontsize: float = 10,
                       figsize: tuple[float, float] = (4,4),
                       return_dataframe: bool = False,
                       return_fig: bool = False,
                       show: bool = True,
                       save: Optional[str] = None
                       ) -> Optional[Union[sns.matrix.ClusterGrid, pd.DataFrame]]:
    """\
    Plot for marker correlation heatmap. 

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
    include_technical_channels
        Whether to include technical channels. If set to False, will exclude
        all channels that are not labeled with `type=="fluo"` in adata.var.
    exclude
        Channels to be excluded from plotting.
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score).
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    corr_method
        correlation method that is used. One of `pearson`, `spearman` or `kendall`.
    cmap
        Sets the colormap for plotting the markers.
    y_label_fontsize
        controls the fontsize of the marker labels.
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    show
        Whether to show the figure. Defaults to True.
    save
        Expects a file path including the file name.
        Saves the figure to the indicated path. Defaults to None.

    Returns
    -------
    If `show==False` a :class:`~seaborn.ClusterGrid`
    If `return_fig==True` a :class:`~seaborn.ClusterGrid`
    If `return_dataframe==True` a :class:`~pandas.DataFrame` containing the data used for plotting

    Examples
    --------
    .. plot::
        :context: close-figs

        import FACSPy as fp

        dataset = fp.mouse_lineages()

        fp.tl.mfi(dataset, layer = "compensated")

        fp.pl.marker_correlation(
            dataset,
            gate = "CD45+",
            layer = "compensated"
        )
    """


    if not isinstance(exclude, list):
        if exclude is None:
            exclude = []
        else:
            exclude = [exclude]

    plot_data = _prepare_heatmap_data(adata = adata,
                                      gate = gate,
                                      layer = layer,
                                      data_metric = data_metric,
                                      data_group = data_group,
                                      include_technical_channels = include_technical_channels,
                                      exclude = exclude,
                                      scaling = scaling)
    
    plot_data = _calculate_correlations(adata = adata,
                                        plot_data = plot_data,
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

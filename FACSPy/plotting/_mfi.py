from anndata import AnnData

from matplotlib.axes import Axes

from typing import Union, Literal, Optional

from ._categorical_stripplot import _categorical_strip_box_plot

from ._utils import (_get_uns_dataframe,
                     savefig_or_show)

from .._utils import _default_gate_and_default_layer, _enable_gate_aliases
from .._settings import settings

@_default_gate_and_default_layer
@_enable_gate_aliases
def fop(adata: AnnData,
        gate: str = None,
        layer: str = None,
        marker: Union[str, list[str]] = None,
        groupby: Union[str, list[str]] = None,
        splitby: str = None,
        cmap: str = None,
        order: list[str] = None,
        stat_test: str = "Kruskal",
        data_group: Optional[Union[str, list[str]]] = "sample_ID",
        data_metric: Literal["mfi", "fop"] = "fop",
        figsize: tuple[float, float] = (3,3),
        return_dataframe: bool = False,
        return_fig: bool = False,
        ax: Axes = None,
        save: bool = None,
        show: bool = None):
    """\
    Plots the fop values as calculated by fp.tl.fop

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
    marker
        The channel to be displayed. Has to be in adata.var_names
    groupby
        controls the x axis and the grouping of the data points
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
    order
        specifies the order of x-values.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots

    Returns
    -------
    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """

    return _mfi_fop_baseplot(adata = adata,
                             gate = gate,
                             data_metric = data_metric,
                             data_group = data_group,
                             layer = layer,
                             marker = marker,
                             groupby = groupby,
                             splitby = splitby,
                             cmap = cmap,
                             figsize = figsize,
                             order = order,
                             stat_test = stat_test,
                             return_fig = return_fig,
                             return_dataframe = return_dataframe,
                             save = save,
                             show = show,
                             ax = ax)


@_default_gate_and_default_layer
@_enable_gate_aliases
def mfi(adata: AnnData,
        gate: str = None,
        layer: str = None,
        marker: Union[str, list[str]] = None,
        groupby: Union[str, list[str]] = None,
        splitby: str = None,
        cmap: str = None,
        order: list[str] = None,
        stat_test: str = "Kruskal",
        data_group: Optional[Union[str, list[str]]] = "sample_ID",
        data_metric: Literal["mfi", "fop"] = "mfi",
        figsize: tuple[float, float] = (3,3),
        return_dataframe: bool = False,
        return_fig: bool = False,
        ax: Axes = None,
        save: bool = None,
        show: bool = None):
    """
    Plots the mfi values as calculated by fp.tl.mfi

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
    marker
        The channel to be displayed. Has to be in adata.var_names
    groupby
        controls the x axis and the grouping of the data points
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
    order
        specifies the order of x-values.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots

    Returns
    -------

    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """
    
    return _mfi_fop_baseplot(adata = adata,
                             gate = gate,
                             data_metric = data_metric,
                             data_group = data_group,
                             layer = layer,
                             marker = marker,
                             groupby = groupby,
                             splitby = splitby,
                             cmap = cmap,
                             figsize = figsize,
                             order = order,
                             stat_test = stat_test,
                             return_fig = return_fig,
                             return_dataframe = return_dataframe,
                             save = save,
                             show = show,
                             ax = ax)

def _mfi_fop_baseplot(adata: AnnData,
                      gate: str,
                      data_metric: str,
                      data_group: str,
                      layer: str,
                      marker: Union[str, list[str]],
                      groupby: Union[str, list[str]],
                      splitby: str,
                      cmap: str = None,
                      stat_test: str = None,
                      order: list[str] = None,
                      figsize: tuple[float, float] = None,
                      return_fig: bool = False,
                      return_dataframe: bool = False,
                      ax: Axes = None,
                      save: bool = None,
                      show: bool = None):
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    if marker is None:
        raise TypeError("Please provide a marker to plot.")
    
    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{data_group}_{layer}")

    if return_dataframe:
        return data
    
    plot_params = {
        "data": data,
        "x": groupby,
        "y": marker,
        "hue": splitby,
        "palette": cmap or settings.default_categorical_cmap if splitby else None,
        "order": order
    }

    fig, ax = _categorical_strip_box_plot(ax = ax,
                                          data = data,
                                          plot_params = plot_params,
                                          groupby = groupby,
                                          splitby = splitby,
                                          stat_test = stat_test,
                                          figsize = figsize)

    ax.set_title(f"{marker}\ngrouped by {groupby}")
    ax.set_xlabel("")
    ax.set_ylabel(f"{marker} {data_metric.upper()} " +
                  f"[{'AFU' if data_metric == 'mfi' else 'dec.'}]")

    if return_fig:
        return fig

    savefig_or_show(save = save, show = show)
    
    if show is False:
        return ax
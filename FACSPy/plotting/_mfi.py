from anndata import AnnData
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from typing import Union, Optional

from ._categorical_stripplot import _categorical_strip_box_plot

from ._utils import (_get_uns_dataframe,
                     savefig_or_show)

from .._utils import _default_gate_and_default_layer, _enable_gate_aliases
from .._settings import settings

@_default_gate_and_default_layer
@_enable_gate_aliases
def fop(adata: AnnData,
        gate: str,
        layer: str,
        marker: str,
        groupby: str,
        splitby: Optional[str] = None,
        cmap: Optional[str] = None,
        order: Optional[Union[list[str], str]] = None,
        stat_test: Optional[str] = "Kruskal",
        data_group: str = "sample_ID",
        figsize: tuple[float, float] = (3,3),
        return_dataframe: bool = False,
        return_fig: bool = False,
        ax: Optional[Axes] = None,
        show: bool = True,
        save: Optional[str] = None
        ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Plots the frequency of parent (fop) values as calculated by fp.tl.fop
    as a combined strip-/boxplot.

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
    splitby
        The parameter controlling additional split along the groupby-axis.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
    order
        specifies the order of x-values.
    stat_test
        Statistical test that is used for the p-value calculation. One of
        `Kruskal` and `Wilcoxon`. Defaults to Kruskal.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    ax
        A :class:`~matplotlib.axes.Axes` created from matplotlib to plot into.
    show
        Whether to show the figure. Defaults to True.
    save
        Expects a file path including the file name.
        Saves the figure to the indicated path. Defaults to None.

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`
    If `return_fig==True` a :class:`~matplotlib.figure.Figure`
    If `return_dataframe==True` a :class:`~pandas.DataFrame` containing the data used for plotting

    Examples
    --------
    .. plot::
        :context: close-figs

        import FACSPy as fp

        dataset = fp.mouse_lineages()
        
        fp.tl.fop(dataset, layer = "compensated")

        fp.pl.fop(
            dataset,
            gate = "CD45+",
            layer = "compensated",
            marker = "B220",
            groupby = "organ",
            figsize = (3,3.5)
        )
    """

    return _mfi_fop_baseplot(adata = adata,
                             gate = gate,
                             data_metric = "fop",
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
        gate: str,
        layer: str,
        marker: str,
        groupby: str,
        splitby: Optional[str] = None,
        cmap: Optional[str] = None,
        order: Optional[Union[list[str], str]] = None,
        stat_test: Optional[str] = "Kruskal",
        data_group: str = "sample_ID",
        figsize: tuple[float, float] = (3,3),
        return_dataframe: bool = False,
        return_fig: bool = False,
        ax: Optional[Axes] = None,
        show: bool = True,
        save: Optional[str] = None
        ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Plots the median fluorescence intensity (mfi) values as calculated by fp.tl.mfi
    as a combined strip-/boxplot.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    marker
        The channel to be displayed. Has to be in adata.var_names.
    groupby
        controls the x axis and the grouping of the data points.
    splitby
        The parameter controlling additional split along the groupby-axis.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
    order
        specifies the order of x-values.
    stat_test
        Statistical test that is used for the p-value calculation. One of
        `Kruskal` and `Wilcoxon`. Defaults to Kruskal.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    ax
        A :class:`~matplotlib.axes.Axes` created from matplotlib to plot into.
    show
        Whether to show the figure. Defaults to True.
    save
        Expects a file path including the file name.
        Saves the figure to the indicated path. Defaults to None.

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`
    If `return_fig==True` a :class:`~matplotlib.figure.Figure`
    If `return_dataframe==True` a :class:`~pandas.DataFrame` containing the data used for plotting

    Examples
    --------
    .. plot::
        :context: close-figs

        import FACSPy as fp

        dataset = fp.mouse_lineages()

        fp.tl.mfi(dataset, layer = "compensated")

        fp.pl.mfi(
            dataset,
            gate = "Neutrophils",
            layer = "compensated",
            marker = "Ly6G",
            groupby = "organ",
            figsize = (3,3.5)
        )
    """
    
    return _mfi_fop_baseplot(adata = adata,
                             gate = gate,
                             data_metric = "mfi",
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
                      marker: str,
                      groupby: str,
                      splitby: Optional[str],
                      cmap: Optional[str],
                      stat_test: Optional[str],
                      order: Optional[Union[list[str], str]],
                      figsize: tuple[float, float],
                      return_fig: bool,
                      return_dataframe: bool,
                      ax: Optional[Axes],
                      save: Optional[str],
                      show: bool):
    
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

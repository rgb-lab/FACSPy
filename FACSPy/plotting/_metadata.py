from matplotlib.axes import Axes
from anndata import AnnData
import pandas as pd

from matplotlib.figure import Figure

from typing import Optional, Union

from ._utils import savefig_or_show
from ._categorical_stripplot import _categorical_strip_box_plot

from .._settings import settings


def metadata(adata: AnnData,
             marker: str,
             groupby: str,
             splitby: Optional[str] = None,
             cmap: Optional[str] = None,
             stat_test: Optional[str] = "Kruskal",
             order: Optional[Union[list[str], str]] = None,
             figsize: tuple[float, float] = (3,3),
             return_dataframe: bool = False,
             return_fig: bool = False,
             ax: Optional[Axes] = None,
             show: bool = True,
             save: Optional[str] = None
             ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Plots the frequency of the metadata columns.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
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
    stat_test
        Statistical test that is used for the p-value calculation. One of
        `Kruskal` and `Wilcoxon`. Defaults to Kruskal.
    order
        specifies the order of x-values.
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

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.pl.metadata(
    ...     dataset,
    ...     groupby = "condition",
    ...     splitby = "sex"
    ... )
    
    """
    
    data = adata.uns["metadata"].dataframe.copy()
    try:
        data[marker] = data[marker].astype("float64")
    except ValueError as e:
        print(str(e))
        if "cast" in str(e) or "convert" in str(e):
            raise ValueError("Please provide a numeric variable for the marker")
        else:
            raise Exception from e

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
    ax.set_ylabel(marker)

    if return_fig:
        return fig    

    savefig_or_show(show = show, save = save)

    if show is False:
        return ax

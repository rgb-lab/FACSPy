from anndata import AnnData
import pandas as pd
from typing import Union

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib import pyplot as plt

from typing import Optional

from ._utils import savefig_or_show
from ._categorical_stripplot import _categorical_strip_box_plot

from .._utils import (_default_gate,
                      subset_gate,
                      _enable_gate_aliases)
from .._settings import settings

def _prep_cluster_abundance(adata: AnnData,
                            groupby: str,
                            cluster_key: str,
                            normalize: bool) -> pd.DataFrame:
    dataframe = adata.obs[[groupby, cluster_key]]
    dataframe = pd.DataFrame(dataframe.groupby(cluster_key).value_counts([groupby]),
                             columns = ["count"]).reset_index()
    if normalize:
        group_sizes = dataframe.groupby(cluster_key).sum("count")
        dataframe = dataframe.groupby([cluster_key, groupby]).mean().div(group_sizes).reset_index()
    
    return dataframe.groupby([cluster_key, groupby], as_index = False)["count"].mean()\
          .pivot(index = cluster_key,
                 columns = groupby,
                 values = "count")

def _order_dataframe(dataframe: pd.DataFrame,
                     order: Union[str, list[str]]) -> pd.DataFrame:
    return dataframe[order]
    
def _prepare_cluster_frequencies(adata: AnnData,
                                 gate: str,
                                 cluster_key: str,
                                 cluster: str,
                                 groupby: str,
                                 splitby: str,
                                 normalize: bool) -> pd.DataFrame:
    adata = subset_gate(adata, gate, as_view = True, copy = False)
    if not splitby:
        groupings = [groupby]
    else:
        groupings = [groupby, splitby]
    if groupby != "sample_ID":
        groupings.append("sample_ID")
    dataframe = adata.obs.groupby(cluster_key).value_counts(groupings).reset_index()
    if normalize:
        cluster_sums = dataframe.groupby(cluster_key).sum("count")
        dataframe = dataframe.set_index(cluster_key)
        normalized_values = dataframe[["count"]] / cluster_sums
        dataframe["count"] = normalized_values["count"]
        dataframe = dataframe.reset_index()
    return dataframe[dataframe[cluster_key] == cluster]


@_default_gate
@_enable_gate_aliases
def cluster_frequency(adata: AnnData,
                      gate: str = None,
                      cluster_key: str = None,
                      cluster: str = None,
                      normalize: bool = False,
                      groupby: Optional[Union[str, list[str]]] = None,
                      splitby: Optional[str] = None,
                      cmap: Optional[str] = None,
                      order: Optional[list[str]] = None,
                      stat_test: str = "Kruskal",
                      figsize: tuple[float, float] = (3,3),
                      return_dataframe: bool = False,
                      return_fig: bool = False,
                      ax: Optional[Axes] = None,
                      show: bool = True,
                      save: Optional[str] = None
                      ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Plots the cluster frequency per cluster as a combined strip-/boxplot.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where Rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    cluster_key
        The `.obs` column where the cluster information is stored.
    cluster
        Specifies the cluster to be analyzed.
    normalize
        If True, normalizes the frequencies to the total amount of cells.
        If False, plots the cell counts per cluster and group.
    groupby
        The parameter to group the x-axis.
    splitby
        The parameter controlling additional split along the groupby-axis.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value.
    order
        specifies the order of x-values.
    stat_test
        Statistical test that is used for the p-value calculation. One of
        `Kruskal` and `Wilcoxon`. Defaults to Kruskal.
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
    If `return_dataframe==True` a :class:`pandas.DataFrame` containg the data used for plotting

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
    >>> fp.tl.leiden(dataset, gate = "T_cells", layer = "transformed")
    >>> fp.pl.cluster_frequency(
    ...     dataset,
    ...     gate = "live",
    ...     cluster_key = "T_cells_transformed_leiden",
    ...     cluster = "2",
    ...     groupby = "condition",
    ...     splitby = "sex",
    ...     normalize = True
    ... )

    """
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    if cluster_key is None:
        raise TypeError("Please provide a cluster key to plot.")
    
    data = _prepare_cluster_frequencies(adata = adata,
                                        gate = gate,
                                        cluster_key = cluster_key,
                                        cluster = cluster,
                                        groupby = groupby,
                                        splitby = splitby,
                                        normalize = normalize)
    if return_dataframe:
        return data

    plot_params = {
        "data": data,
        "x": groupby,
        "y": "count",
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

    ax.set_title(f"Cluster frequency\nper {groupby}")
    ax.set_xlabel("")
    ax.set_ylabel(f"cluster frequency [{'dec' if normalize else 'count'}]")

    if return_fig:
        return fig

    savefig_or_show(save = save, show = show)
    
    if show is False:
        return ax    

    return

def cluster_abundance(adata: AnnData,
                      groupby: Union[str, list[str]],
                      cluster_key: str = None,
                      normalize: bool = True,
                      order: Optional[list[str]] = None,
                      figsize: tuple[float, float] = (5,4),
                      return_dataframe: bool = False,
                      return_fig: bool = False,
                      ax: Optional[Axes] = None,
                      show: bool = True,
                      save: Optional[str] = None
                      ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Plots the frequency as a stacked bar chart of a grouping variable per cluster.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    groupby
        controls the x axis and the grouping of the data points.
    cluster_key
        The obs slot where the clusters of interest are stored.
    normalize
        If True, normalizes the frequencies to 1. If False, the y-axis
        represents the cell counts per cluster.
    order
        Sets the order of the groupby variable.
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    ax
        A :class:`~matplotlib.axes.Axes` to created from matplotlib to plot into.
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
    >>> fp.tl.leiden(dataset, gate = "T_cells", layer = "transformed")
    >>> fp.pl.cluster_abundance(
    ...     dataset,
    ...     gate = "live",
    ...     cluster_key = "T_cells_transformed_leiden",
    ...     groupby = "condition",
    ...     normalize = True
    ... )


    """
    
    dataframe = _prep_cluster_abundance(adata, groupby, cluster_key, normalize)
    
    if order is not None:
        dataframe = _order_dataframe(dataframe, order)

    if return_dataframe:
        return dataframe
    
    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)
    dataframe.plot(kind = "bar",
                   stacked = True,
                   ax = ax)
    ax.legend(loc = "upper left",
              bbox_to_anchor = (1,1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_title(f"{cluster_key} cluster frequency\nper {groupby}")
    ax.set_ylabel("frequency [dec.]")

    if return_fig:
        return fig

    savefig_or_show(save = save, show = show)

    if show is False:
        return ax
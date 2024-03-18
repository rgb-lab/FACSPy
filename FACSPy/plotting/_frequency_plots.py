from anndata import AnnData
import pandas as pd
from typing import Union

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib import pyplot as plt

from typing import Optional

from ._utils import savefig_or_show
from ._categorical_stripplot import _categorical_strip_box_plot

from .._utils import (_default_gate_and_default_layer,
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


@_default_gate_and_default_layer
@_enable_gate_aliases
def cluster_frequency(adata: AnnData,
                      gate: str = None,
                      layer: str = None,
                      cluster_key: Union[str, list[str]] = None,
                      cluster: str = None,
                      normalize: bool = False,
                      groupby: Union[str, list[str]] = None,
                      splitby: str = None,
                      cmap: str = None,
                      order: list[str] = None,
                      stat_test: str = "Kruskal",
                      figsize: tuple[float, float] = (3,3),
                      return_dataframe: bool = False,
                      return_fig: bool = False,
                      ax: Axes = None,
                      save: bool = None,
                      show: bool = None):
    
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
                      ax: Axes = None,
                      return_dataframe: bool = False,
                      return_fig: bool = False,
                      save: bool = None,
                      show: bool = None) -> Optional[Figure]:
    """\
    Plots the frequency as a stacked bar chart of a grouping variable per cluster.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    groupby
        controls the x axis and the grouping of the data points
    cluster_key
        The obs slot where the clusters of interest are stored
    normalize
        Whether to normalize the frequencies to 1.
    order
        Sets the order of the groupby variable.
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
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots

    Returns
    -------
    if `show==False` a :class:`~seaborn.ClusterGrid`

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
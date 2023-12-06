from anndata import AnnData
import pandas as pd
from typing import Union

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib import pyplot as plt

from typing import Optional

from ._utils import savefig_or_show

def _prep_dataframe_cluster_freq(adata: AnnData,
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

def cluster_frequency(adata: AnnData,
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
    """
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

    
    
    dataframe = _prep_dataframe_cluster_freq(adata, groupby, cluster_key, normalize)
    
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
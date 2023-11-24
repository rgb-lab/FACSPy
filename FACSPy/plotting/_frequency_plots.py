from anndata import AnnData
import pandas as pd
from typing import Union

from matplotlib.figure import Figure

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
                      order: Optional[list[str]] = None,
                      cluster_key: str = "leiden",
                      normalize: bool = True,
                      return_dataframe: bool = False,
                      return_fig: bool = False,
                      save: bool = None,
                      show: bool = None) -> Optional[Figure]:
    
    dataframe = _prep_dataframe_cluster_freq(adata, groupby, cluster_key, normalize)
    
    if order is not None:
        dataframe = _order_dataframe(dataframe, order)

    if return_dataframe:
        return dataframe
    
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5,4))
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
    plt.tight_layout()

    savefig_or_show(save = save, show = show)
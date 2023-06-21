from anndata import AnnData
import pandas as pd
import numpy as np
from typing import Union, Literal

from matplotlib.figure import Figure
from matplotlib.axis import Axis

from matplotlib import pyplot as plt
import seaborn as sns

from ..utils import find_gate_path_of_gate

from ..exceptions.exceptions import AnalysisNotPerformedError

from .utils import create_boxplot, append_metadata, turn_off_missing_plots, prep_uns_dataframe

from typing import Optional

def prep_dataframe_cluster_freq(adata: AnnData,
                                groupby: str,
                                cluster_key: str,
                                normalize: bool) -> pd.DataFrame:
    dataframe = adata.obs[[groupby, cluster_key]]
    dataframe = pd.DataFrame(dataframe.groupby(cluster_key).value_counts([groupby]),
                             columns = ["count"]).reset_index()
    if normalize:
        group_sizes = dataframe.groupby(cluster_key).sum()
        dataframe = dataframe.groupby([cluster_key, groupby]).mean().div(group_sizes).reset_index()
    
    return dataframe.groupby([cluster_key, "condition"], as_index = False)["count"].mean()\
        .pivot(index = cluster_key,
               columns = groupby,
               values = "count")


def cluster_frequency(adata: AnnData,
                      groupby: Union[str, list[str]],
                      cluster_key: str = "leiden",
                      normalize: bool = True) -> Optional[Figure]:
    
    dataframe = prep_dataframe_cluster_freq(adata, groupby, cluster_key, normalize)
    print(dataframe.shape)
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5,4))
    dataframe.plot(kind = "bar",
                   stacked = True,
                   ax = ax)
    ax.legend(loc = "upper left",
              bbox_to_anchor = (1,1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    ax.set_title(f"{cluster_key} cluster frequency\nper {groupby}")
    plt.tight_layout()
    plt.show()
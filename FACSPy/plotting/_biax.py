import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure

from anndata import AnnData

from typing import Union, Optional, Literal


def biax(adata: AnnData,
         x_channel: str,
         y_channel: str,
         sample_ID: Union[str, list[str]], 
         layer: Literal["compensated", "raw", "transformed"] = "compensated",
         add_cofactor: bool = False) -> Optional[Figure]:
    
    if not isinstance(sample_ID, list):
        sample_ID = [sample_ID]
    dataframe = adata[adata.obs["sample_ID"].isin(sample_ID)].to_df(layer = layer)
    dataframe["sample_ID"] = adata[adata.obs["sample_ID"].isin(sample_ID)].obs["sample_ID"].to_list()

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5,5))
    plot_params = {
        "data": dataframe,
        "x": x_channel,
        "y": y_channel,
        "linewidth": 0,
        "s": 2,
        "c": dataframe["sample_ID"]
    }
    sns.scatterplot(**plot_params,
                    ax = ax)

    if layer == "compensated":
        plt.xscale("symlog", linthresh = 5)
        plt.yscale("symlog", linthresh = 5)
    plt.tight_layout()
    plt.show()

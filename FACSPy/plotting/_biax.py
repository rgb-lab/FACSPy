import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from anndata import AnnData

from typing import Union, Optional, Literal

from .utils import savefig_or_show

#TODO: add legend
def biax(adata: AnnData,
         x_channel: str,
         y_channel: str,
         sample_ID: Union[str, list[str]], 
         layer: Literal["compensated", "raw", "transformed"] = "compensated",
         add_cofactor: bool = False,
         show: Optional[bool] = None,
         save: Optional[Union[str, bool]] = None,
         return_dataframe: bool = False,
         return_fig: bool = False) -> Optional[Figure]:
    
    if not isinstance(sample_ID, list):
        sample_ID = [sample_ID]
    
    for sample in sample_ID:
        if sample not in adata.obs["sample_ID"].unique():
            raise ValueError(f"sampleID {sample} not found!")
    
    dataframe = adata[adata.obs["sample_ID"].isin(sample_ID)].to_df(layer = layer)
    dataframe["sample_ID"] = adata[adata.obs["sample_ID"].isin(sample_ID)].obs["sample_ID"].to_list()

    if return_dataframe:
        return dataframe

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (3,3))
    plot_params = {
        "data": dataframe,
        "x": x_channel,
        "y": y_channel,
        "linewidth": 0,
        "s": 2,
        "c": dataframe["sample_ID"],
        "legend": "auto"
    }
    sns.scatterplot(**plot_params,
                    ax = ax)

    if layer in ["compensated", "raw"]:
        x_channel_cofactor = float(adata.var.loc[adata.var["pns"] == x_channel, "cofactors"].iloc[0])
        y_channel_cofactor = float(adata.var.loc[adata.var["pns"] == y_channel, "cofactors"].iloc[0])
        plt.xscale("symlog",
                   linthresh = x_channel_cofactor)
        ax.axvline(x_channel_cofactor)
        plt.yscale("symlog",
                   linthresh = y_channel_cofactor)
        ax.axhline(y_channel_cofactor)
    if layer in ["transformed"]:
        ax.axhline(np.arcsinh(1))
        ax.axvline(np.arcsinh(1))
    ax.set_title(f"{layer}\nexpression")
    plt.tight_layout()

    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.figure import Figure

from anndata import AnnData

from typing import Union, Optional, Literal

from ._utils import savefig_or_show
from .._utils import (subset_gate,
                      is_valid_filename,
                      is_valid_sample_ID,
                      _default_layer)

#TODO: add legend
@_default_layer
def biax(adata: AnnData,
         gate: str,
         x_channel: str,
         y_channel: str,
         sample_identifier: Union[str, list[str]], 
         layer: Literal["compensated", "raw", "transformed"] = "compensated",
         add_cofactor: bool = False,
         show: Optional[bool] = None,
         save: Optional[Union[str, bool]] = None,
         return_dataframe: bool = False,
         return_fig: bool = False) -> Optional[Figure]:
    
    
    adata = subset_gate(adata, gate = gate, as_view = True)
    if is_valid_sample_ID(adata, sample_identifier):
        sample_specific = adata[adata.obs["sample_ID"] == str(sample_identifier),:]
    elif is_valid_filename(adata, sample_identifier):
        sample_specific = adata[adata.obs["file_name"] == str(sample_identifier),:]
    else:
        raise ValueError(f"{sample_identifier} not found")
    
    expression_data = sample_specific.to_df(layer = layer)
    obs_data = sample_specific.obs.copy()
    dataframe = pd.concat([expression_data, obs_data], axis = 1)

    if return_dataframe:
        return dataframe

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (3,3))
    plot_params = {
        "data": dataframe,
        "x": x_channel,
        "y": y_channel,
        "linewidth": 0,
        "s": 2,
        "c": dataframe["sample_ID"].astype(int),
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

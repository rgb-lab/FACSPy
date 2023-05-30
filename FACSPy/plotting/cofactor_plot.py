from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from ..exceptions.exceptions import CofactorsNotCalculatedError

from .utils import calculate_fig_size, turn_off_missing_plots, calculate_nrows

from scanpy.plotting._utils import savefig_or_show

from typing import Optional, Union


def transformation_plot(adata: AnnData,
                        sample_ID: Optional[str],
                        file_name: Optional[str],
                        ):
    
    pass

def cofactor_distribution(adata: AnnData,
                          groupby: Optional[str] = None,
                          channel: Optional[str] = None,
                          return_fig: Optional[bool] = False,
                          ax: Optional[Axes] = None,
                          ncols: int = 4,
                          show: bool = True) -> Union[Figure, Axes, None]:
    
    assert "raw_cofactors" in adata.uns, "raw cofactors not found..."
    cofactors = adata.uns["raw_cofactors"]
    metadata = adata.uns["metadata"].to_df()
    data = cofactors.merge(metadata, left_index = True, right_on = "file_name")
    
    nrows = calculate_nrows(ncols, cofactors)
    figsize = calculate_fig_size(ncols,
                                 nrows,
                                 data[groupby].unique() if groupby else None)
    
    fig, ax = plt.subplots(ncols = ncols,
                           nrows = nrows,
                           figsize = figsize)
    
    ax = np.ravel(ax)
    
    for i, marker in enumerate(cofactors.columns):
        plot_params = {
            "y": marker,
            "x": groupby,
            "data": data
        }
        sns.boxplot(boxprops = dict(facecolor = "white"),
                    ax = ax[i],
                    whis = (0,100),
                    **plot_params)
        sns.stripplot(linewidth = 1,
                      dodge = True,
                      ax = ax[i],
                      **plot_params)
        ax[i].set_title(marker, fontsize = 10)
    
    ax: list[Axes] = turn_off_missing_plots(ax)
    
    for axs in ax:
        axs.set_ylim(0, axs.get_ylim()[1] * 1.2)
        axs.set_ylabel("AFU")
        axs.set_xticklabels(axs.get_xticklabels(), rotation = 45, ha = "right")
        axs.set_xlabel("")
    
    ax = np.reshape(ax, (ncols, nrows))
    
    if return_fig:
        return fig
    
    if not show:
        return ax
    
    plt.tight_layout()
    plt.show()
    



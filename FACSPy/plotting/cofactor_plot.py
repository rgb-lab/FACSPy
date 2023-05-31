from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from ..exceptions.exceptions import CofactorsNotCalculatedError

from .utils import calculate_fig_size, turn_off_missing_plots, calculate_nrows

from scanpy.plotting._utils import savefig_or_show

from typing import Optional, Union, Literal

from ..dataset.utils import find_corresponding_control_samples, create_sample_subset_with_controls
from ..utils import subset_gate


def prepare_data_subsets(adata: AnnData,
                         by: Literal["sample_ID", "file_name"],
                         sample_identifier: str,
                         marker: str,
                         scatter: str,
                         sample_size: int = 5_000)-> tuple[AnnData, AnnData]:

    (_,
     corresponding_control_samples) = find_corresponding_control_samples(adata,
                                                                         by = by)

    stained_sample = adata[adata.obs[by] == sample_identifier,
                           adata.var_names.isin([scatter, marker])]
    control_samples = adata[adata.obs[by].isin(corresponding_control_samples[sample_identifier]),
                            adata.var_names.isin([scatter, marker])]
    
    if stained_sample.shape[0] > sample_size:
        sc.pp.subsample(stained_sample, n_obs = sample_size)
    if control_samples.shape[0] > sample_size:
        sc.pp.subsample(control_samples, n_obs = sample_size)
    return stained_sample, control_samples

def calculate_plot_limits(stained_data: pd.DataFrame,
                          control_data: pd.DataFrame,
                          y_channel: str) -> tuple[int, int]:

    if control_data.shape[0] > 0:
        combined = pd.concat([stained_data, control_data], axis = 0)
    else:
        combined = stained_data
    return (0, combined[y_channel].max() * 1.1)


def transformation_scatter_plot(type: Literal["compensated", "transformed"],
                                ax: Axes,
                                stained_data,
                                control_data,
                                cofactor,
                                scatter,
                                plot_params: dict) -> Axes:
    control_is_present = control_data.shape[0] > 0
    ax = sns.scatterplot(data = stained_data,
                         ax = ax,
                         color = "red",
                         **plot_params)
    if control_is_present:
        sns.scatterplot(data = control_data,
                        ax = ax,
                        color = "blue",
                        **plot_params)

    if type == "compensated":
        ax.set_xscale("symlog", linthresh = cofactor)

    ymin, ymax = calculate_plot_limits(stained_data,
                                       control_data,
                                       scatter)
    
    ax.axvline(x = cofactor if type == "compensated" else np.arcsinh(1),
               ymin = ymin,
               ymax = ymax,
               color = "green")
    ax.set_ylim(ymin, ymax)
    
    return ax

def transformation_plot(adata: AnnData,
                        sample_ID: Optional[str] = None,
                        file_name: Optional[str] = None,
                        pregated_population: Optional[str] = None,
                        marker: Optional[str] = None,
                        scatter: str = "SSC-A",
                        sample_size: Optional[int] = 5_000
                        ):

    
    if sample_ID and file_name:
        raise TypeError("Please provide one of sample_ID or file_name but not both.")
    
    if pregated_population:
        ### TODO TODO TODO: CHECK IF POPULATION EXISTS!!
        adata = subset_gate(adata,
                            gate = pregated_population,
                            as_view = True)
    
    stained_sample, control_samples = prepare_data_subsets(adata,
                                                           by = "sample_ID" if sample_ID else "file_name",
                                                           sample_identifier = sample_ID or file_name,
                                                           marker = marker,
                                                           scatter = scatter,
                                                           sample_size = sample_size)

    cofactor = adata.uns["cofactors"].get_cofactor(marker)
    
    fig: Figure
    ax: list[Axes]
    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (15,5))
    
    plot_params = {
        "x": marker,
        "y": scatter,
        "s": 1,
        "linewidth": 0
    }
    ### first plot: raw fluorescence values
    
    stained_data = stained_sample.to_df(layer = "compensated")
    control_data = control_samples.to_df(layer = "compensated")
    ax[0] = transformation_scatter_plot(type = "compensated",
                                        ax = ax[0],
                                        stained_data = stained_data,
                                        control_data = control_data,
                                        cofactor = cofactor,
                                        scatter = scatter,
                                        plot_params = plot_params)
    
    stained_data = stained_sample.to_df(layer = "transformed")
    control_data = control_samples.to_df(layer = "transformed")
    ax[1] = transformation_scatter_plot(type = "transformed",
                                        ax = ax[1],
                                        stained_data = stained_data,
                                        control_data = control_data,
                                        cofactor = cofactor,
                                        scatter = scatter,
                                        plot_params = plot_params)
    

    plt.show()
    


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
    



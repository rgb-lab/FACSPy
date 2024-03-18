from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._utils import (calculate_fig_size,
                    turn_off_missing_plots,
                    calculate_nrows,
                    savefig_or_show)

from typing import Optional, Union, Literal

from ..dataset._utils import (find_corresponding_control_samples,
                              _get_histogram_curve)
from .._utils import (subset_gate,
                      _is_valid_sample_ID,
                      _is_valid_filename,
                      _default_gate,
                      _enable_gate_aliases)


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

def combine_data_if_control_is_present(df1: pd.DataFrame,
                                       df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1, df2], axis = 0) if df2.shape[0] > 0 else df1

def calculate_y_plot_limits(stained_data: pd.DataFrame,
                          control_data: pd.DataFrame,
                          y_channel: str) -> tuple[int, int]:

    combined = combine_data_if_control_is_present(stained_data, control_data)
    return (combined[y_channel].quantile(0.001), combined[y_channel].max())

def calculate_x_plot_limits(stained_data: pd.DataFrame,
                            control_data: pd.DataFrame,
                            x_channel: str) -> tuple[int, int]:
    combined = combine_data_if_control_is_present(stained_data, control_data)
    return (combined[x_channel].quantile(0.001), combined[x_channel].max())

def normalize_histogram_curve(curve: np.ndarray) -> np.ndarray:
    curve *= (1/np.max(curve))
    return curve    

def calculate_histogram_data(data: pd.DataFrame,
                             plot_params: dict) -> tuple[np.ndarray, np.ndarray]:
    x, curve = _get_histogram_curve(data[plot_params["x"]].values)
    curve = normalize_histogram_curve(curve)
    return x[:100], curve

def append_cofactor_label(ax: Axes,
                          x: float,
                          y: float) -> Axes:
    ax.text(x = x,
            y = y,
            s = "Cofactor",
            fontdict = {
                "size": 10,
                "weight": "bold"
            })
    return ax

def transformation_scatter_plot(type: Literal["compensated", "transformed"],
                                ax: Axes,
                                stained_data,
                                control_data,
                                cofactor,
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

    ymin, ymax = calculate_y_plot_limits(stained_data,
                                         control_data,
                                         plot_params["y"])
    xmin, xmax = calculate_x_plot_limits(stained_data,
                                         control_data,
                                         plot_params["x"])
    ax.axvline(x = cofactor if type == "compensated" else np.arcsinh(1),
               color = "green")
    ax = append_cofactor_label(ax = ax,
                               x = cofactor * 1.05 if type == "compensated" else 1,
                               y = ymin + (ymax - ymin) * 0.9)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_title(f"Compensated Values\nScatter Plot - {plot_params['x']}"
                 if type == "compensated"
                 else f"Transformed Values\nScatter Plot - {plot_params['x']}")
    
    return ax


def transformation_histogram_plot(type: Literal["compensated", "transformed"],
                                  ax: Axes,
                                  stained_data: pd.DataFrame,
                                  stained_data_x: np.ndarray,
                                  stained_data_curve: np.ndarray,
                                  control_data: pd.DataFrame,
                                  control_data_x: np.ndarray,
                                  control_data_curve: np.ndarray,
                                  plot_params: dict) -> Axes:
    control_is_present = control_data_x is not None and control_data_curve is not None
    
    ax = sns.lineplot(x = stained_data_x,
                      y = stained_data_curve,
                      color = "red",
                      label = "stained",
                      ax = ax)
    if control_is_present:
        ax = sns.lineplot(x = control_data_x,
                          y = control_data_curve,
                          color = "blue",
                          label = "control",
                          ax = ax)
    xmin, xmax = calculate_x_plot_limits(stained_data,
                                         control_data,
                                         plot_params["x"])
    ax.set_ylim(-0.02, 1.02)
    ax.axvline(x = np.arcsinh(1),
               color = "green")
    ax = append_cofactor_label(ax = ax,
                               x = 1,
                               y = 0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_title(f"Transformed Values\nHistogram - {plot_params['x']}")
    return ax

@_default_gate
@_enable_gate_aliases
def transformation_plot(adata: AnnData,
                        gate: Optional[str] = None,
                        sample_identifier: str = None,
                        marker: Optional[Union[str, list[str]]] = None,
                        scatter: str = "SSC-A",
                        sample_size: Optional[int] = 5_000,
                        figsize: tuple[float, float] = (10,3),
                        return_dataframe: bool = False,
                        return_fig: bool = False,
                        save: bool = None,
                        show: bool = None
                        ) -> Union[Figure, Axes, tuple[pd.DataFrame]]:
    """\
    Transformation plot. Plots the data on a log scale (biaxial), the data on the transformed scale (biaxial) and the data on a transformed scale as histogram.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    sample_identifier
        Used to specify a specific sample. Can be a valid sample_ID and a valid file_name
        from the .obs slot or the metadata
    marker
        the channel to plot
    scatter
        the scatter channel to use on the y-axis. Defaults to SSC-A
    sample_size
        Controls how many data points are displayed. Defaults to 5000. More displayed data
        points can significantly increase plotting time.
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.

    Returns
    -------
    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """

    if not _is_valid_sample_ID(adata, sample_identifier) and not _is_valid_filename(adata, sample_identifier):
        raise ValueError(f"{sample_identifier} not found")
  
    if gate:
        adata = subset_gate(adata,
                            gate = gate,
                            as_view = True)
       
    stained_sample, control_samples = prepare_data_subsets(adata,
                                                           by = "sample_ID" if _is_valid_sample_ID(adata, sample_identifier) else "file_name",
                                                           sample_identifier = sample_identifier,
                                                           marker = marker,
                                                           scatter = scatter,
                                                           sample_size = sample_size)

    cofactor = adata.uns["cofactors"].get_cofactor(marker)

    plot_params = {
        "x": marker,
        "y": scatter,
        "s": 3,
        "linewidth": 0.1
    }
    ### first plot: raw fluorescence values
    
    stained_data_compensated = stained_sample.to_df(layer = "compensated")
    control_data_compensated = control_samples.to_df(layer = "compensated")
    
    stained_data_transformed = stained_sample.to_df(layer = "transformed")
    control_data_transformed = control_samples.to_df(layer = "transformed")

    stained_data_x, stained_data_curve = calculate_histogram_data(stained_data_transformed,
                                                                  plot_params = plot_params)
    if control_data_transformed.shape[0] > 0:
        control_data_x, control_data_curve = calculate_histogram_data(control_data_transformed,
                                                                      plot_params = plot_params)
    else:
        control_data_x, control_data_curve = None, None

    if return_dataframe:
        return (stained_data_compensated,
                stained_data_transformed,
                control_data_compensated,
                control_data_transformed,
                stained_data_x,
                stained_data_curve,
                control_data_x,
                control_data_curve)
    
    fig: Figure
    ax: list[Axes]
    ncols = 3
    nrows = 1
    figsize = (10, 3)
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
    ax[0] = transformation_scatter_plot(type = "compensated",
                                        ax = ax[0],
                                        stained_data = stained_data_compensated,
                                        control_data = control_data_compensated,
                                        cofactor = cofactor,
                                        plot_params = plot_params)
    
    ax[1] = transformation_scatter_plot(type = "transformed",
                                        ax = ax[1],
                                        stained_data = stained_data_transformed,
                                        control_data = control_data_transformed,
                                        cofactor = cofactor,
                                        plot_params = plot_params)

    ax[2] = transformation_histogram_plot(type = "transformed",
                                          ax = ax[2],
                                          stained_data = stained_data_transformed,
                                          stained_data_x = stained_data_x,
                                          stained_data_curve = stained_data_curve,
                                          control_data = control_data_transformed,
                                          control_data_x = control_data_x,
                                          control_data_curve = control_data_curve,
                                          plot_params = plot_params)

    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles,
                 labels,
                 loc = "center left",
                 bbox_to_anchor = (1.1, 0.5))

    if return_fig:
        return fig
    
    savefig_or_show(show = show, save = save)

    if show is False:
        return ax
    

def cofactor_distribution(adata: AnnData,
                          groupby: Optional[str] = None,
                          channels: Optional[str] = None,
                          ncols: int = 4,
                          return_dataframe: bool = False,
                          return_fig: bool = False,
                          save: bool = None,
                          show: bool = None) -> Union[Figure, Axes, None]:
    """
    Plots the cofactor distribution of specific channels

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    groupby
        controls the x-axis and sets the variables to group the data by. Should be categorical.
    channels
        the channels to display. Each channel gets its individual plot.
    ncols
        controls the number of columns for the plotting of multiple channels
    show
        whether to show the figure
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.

    Returns
    -------

    if `show==False` a :class:`~matplotlib.axes.Axes`
 
    """
    
    assert "raw_cofactors" in adata.uns, "raw cofactors not found..."
    cofactors = adata.uns["raw_cofactors"]
    if channels:
        if not isinstance(channels, list):
            channels = [channels]
        cofactors = cofactors.loc[:, cofactors.columns.isin(channels)]
    metadata = adata.uns["metadata"].to_df()
    data = cofactors.merge(metadata, left_index = True, right_on = "file_name")
    if return_dataframe:
        return data

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
    
    savefig_or_show(show = show, save = save)

    if show is False:
        return ax
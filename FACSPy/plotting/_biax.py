import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from matplotlib.figure import Figure

from anndata import AnnData

from typing import Union, Optional, Literal
from scipy.interpolate import interpn
from ._utils import (savefig_or_show,
                     _color_var_is_categorical,
                     _continous_color_vector,
                     _retrieve_cofactor_or_set_to_default,
                     _generate_continous_color_scale,
                     _define_axis_scale,
                     _transform_data_to_scale,
                     _transform_color_to_scale)
from .._utils import (subset_gate,
                      is_valid_filename,
                      is_valid_sample_ID,
                      _default_layer)

def _generate_scale_kwargs(channel,
                           channel_scale,
                           linthresh: float) -> dict:
    channel_scale: Literal["linear", "log", "symlog"]= _define_axis_scale(channel, channel_scale)
    scale_kwargs = {
        "value": channel_scale 
    }
    if channel_scale == "symlog":
        scale_kwargs["linthresh"] = linthresh
    return scale_kwargs
    
def _get_cmap_biax(cmap,
                   color):
    if cmap:
        return cmap
    if color == "density":
        return "jet"
    return "viridis"

def _create_expression_frame(adata: AnnData,
                             layer: str) -> pd.DataFrame:
    expression_data = adata.to_df(layer = layer)
    obs_data = adata.obs.copy()
    return pd.concat([expression_data, obs_data], axis = 1)

def _calculate_density(x: np.ndarray,
                       y: np.ndarray,
                       bins: float = 20) -> np.ndarray:
    ## https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density
    data, x_e, y_e = np.histogram2d(x, y, bins = bins, density = True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]),
                 0.5*(y_e[1:]+y_e[:-1])),
                 data,
                 np.vstack([x,y]).T,
                 method = "splinef2d",
                 bounds_error = False)
    z[np.where(np.isnan(z))] = 0.0
    return z

@_default_layer
def biax(adata: AnnData,
         gate: str,
         layer: Optional[str] = None,
         x_channel: str = None,
         y_channel: str = None,
         color: Optional[Union[str, Literal["density"]]] = "density",
         sample_identifier: Union[str, list[str]] = None, 
         add_cofactor: Literal["x", "y", "both"] = False,
         x_scale: Literal["biex", "log", "linear"] = None,
         y_scale: Literal["biex", "log", "linear"] = None,
         color_scale: Literal["biex", "log", "linear"] = "linear",
         cmap: str = None,
         vmin: float = None,
         vmax: float = None,
         figsize: tuple[float, float] = (4,4),
         title: Optional[str] = None,
         show: Optional[bool] = None,
         save: Optional[Union[str, bool]] = None,
         ax: Axes = None,
         return_dataframe: bool = False,
         return_fig: bool = False) -> Optional[Figure]:
    
    """
    Plot for normal biaxial representation of cytometry data.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    x_channel
        The channel that is plotted on the x axis.
    y_channel
        The channel that is plotted on the y axis
    color
        The parameter that controls the coloring of the plot.
        Can be set to categorical variables from the .obs slot
        or continuous variables corresponding to channels.
        Default is set to 'density', which calculates the point
        density in the plot.
    sample_identifier
        Controls the data that are extracted. Defaults to None.
        If set, has to be one of the sample_IDs or the file_names.
    add_cofactor
        if set, adds the cofactor as a line to the plot for visualization.
        if 'x', sets the cofactor for the x-axis,
        if 'y', sets the cofactor for the y-axis,
        if 'both', sets both axis cofactors
    x_scale
        sets the scale for the x axis. Has to be one of 'biex', 'log', 'linear'.
        The value 'biex' gets converted to 'symlog' internally
    y_scale
        sets the scale for the y axis. Has to be one of 'biex', 'log', 'linear'.
        The value 'biex' gets converted to 'symlog' internally
    color_scale
        sets the scale for the colorbar. Has to be one of 'biex', 'log', 'linear'.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
    vmin
        minimum value to plot in the color vector
    vmax
        maximum value to plot in the color vector
    figsize
        contains the dimensions of the final figure as a tuple of two ints or floats
    title
        sets the figure title. Optional
    show
        whether to show the fig
    save
        expects a file path and a file name. saves the figure to the indicated path
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots

    Returns
    -------

    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """
    
    if x_channel is None or y_channel is None:
        raise ValueError("Please provide x_channel and y_channel")
    if x_scale not in ["biex", "linear", "log"] and x_scale is not None:
        raise ValueError("parameter x_scale has to be one of ['biex', 'linear', 'log']")
    if y_scale not in ["biex", "linear", "log"] and y_scale is not None:
        raise ValueError("parameter x_scale has to be one of ['biex', 'linear', 'log']")
    if color_scale not in ["biex", "linear", "log"] and color_scale is not None:
        raise ValueError("parameter x_scale has to be one of ['biex', 'linear', 'log']")
    
    adata = subset_gate(adata, gate = gate, as_view = True)
    if sample_identifier is not None:
        if is_valid_sample_ID(adata, sample_identifier):
            adata = adata[adata.obs["sample_ID"] == str(sample_identifier),:]
        elif is_valid_filename(adata, sample_identifier):
            adata = adata[adata.obs["file_name"] == str(sample_identifier),:]
        else:
            raise ValueError(f"{sample_identifier} not found")

    dataframe = _create_expression_frame(adata,
                                         layer)    

    if color == "density":
        # we need to know how to plot as this will affect the
        # visible density of data points
        x = _transform_data_to_scale(dataframe[x_channel].values,
                                     x_channel,
                                     adata,
                                     x_scale)
        y = _transform_data_to_scale(dataframe[y_channel].values,
                                     y_channel,
                                     adata,
                                     y_scale)

        dataframe["density"] = _calculate_density(x = x, y = y)

        # also, we set vmin and vmax to None as there is no colorbar
        vmin = None
        vmax = None
        # color scale is still set explicitly
        color_scale = "linear"

    if return_dataframe:
        return dataframe
    
    ### if plotting categorical colors with multiple sample
    ### this step is necessary to shuffle the colors
    dataframe = dataframe.sample(frac = 1)

    categorical_color = _color_var_is_categorical(dataframe[color])
    if not categorical_color:
        if color in adata.var_names and color_scale == "biex":
            color_cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                                  color)
        else:
            color_cofactor = None

        color_vector = _continous_color_vector(dataframe,
                                               color,
                                               vmin,
                                               vmax)
        transformed_color_vector= _transform_color_to_scale(color_vector,
                                                            color_cofactor,
                                                            color_scale)

    continous_cmap = _get_cmap_biax(cmap, color)

    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)

    plot_params = {
        "data": dataframe,
        "x": x_channel,
        "y": y_channel,
        "linewidth": 0,
        "s": 2,
        "hue": dataframe[color] if categorical_color else None,
        "palette": cmap or "Set1",
        "c": transformed_color_vector if not categorical_color else None,
        "cmap": continous_cmap,
        "legend": "auto"
    }

    sns.scatterplot(**plot_params,
                    ax = ax)

    ### axis scaling:
    x_channel_cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                              x_channel)

    y_channel_cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                              y_channel)
    
    x_scale_kwargs = _generate_scale_kwargs(x_channel,
                                            x_scale,
                                            x_channel_cofactor)
    y_scale_kwargs = _generate_scale_kwargs(y_channel,
                                            y_scale,
                                            y_channel_cofactor)
    ax.set_xscale(**x_scale_kwargs)
    ax.set_yscale(**y_scale_kwargs)

    if layer in ["compensated", "raw"] and add_cofactor:
        if add_cofactor == "x" or add_cofactor == "both":
            ax.axvline(x_channel_cofactor)
        if add_cofactor == "y" or add_cofactor == "both":
            ax.axhline(y_channel_cofactor)
    if layer in ["transformed"] and add_cofactor:
        if add_cofactor == "x" or add_cofactor == "both":
            ax.axhline(np.arcsinh(0.88))
        if add_cofactor == "y" or add_cofactor == "both":
            ax.axvline(np.arcsinh(0.88))

    if title:
        ax.set_title(title)
    if categorical_color:
        ax.legend(bbox_to_anchor = (1.1, 0.5), loc = "center left")
    if not categorical_color and color != "density":
        cbar = _generate_continous_color_scale(color_vector = color_vector,
                                               cmap = continous_cmap,
                                               color_cofactor = color_cofactor,
                                               ax = ax,
                                               color_scale = color_scale)
        cbar.ax.set_ylabel(f"{layer} expression\n{color}",
                           rotation = 270,
                           labelpad = 30)
    plt.tight_layout()

    if return_fig:
        return ax

    savefig_or_show(show = show, save = save)

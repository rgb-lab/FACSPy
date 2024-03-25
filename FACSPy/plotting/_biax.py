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
                     _generate_scale_kwargs,
                     _transform_data_to_scale,
                     _transform_color_to_scale)
from .._utils import (subset_gate,
                      _is_valid_filename,
                      _is_valid_sample_ID,
                      _default_gate_and_default_layer,
                      _enable_gate_aliases)

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
                       bins: int = 20) -> np.ndarray:
    ## https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density
    data, x_e, y_e = np.histogram2d(x, y, bins = bins, density = True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]),
                 0.5*(y_e[1:]+y_e[:-1])),
                 data,
                 np.vstack([x,y]).T,
                 method = "splinef2d",
                 bounds_error = False)
    assert z is not None
    z[np.where(np.isnan(z))] = 0.0
    return z

@_default_gate_and_default_layer
@_enable_gate_aliases
def biax(adata: AnnData,
         gate: str,
         layer: str,
         x_channel: str,
         y_channel: str,
         sample_identifier: Optional[str] = None, 
         color: Optional[Union[str, Literal["density"]]] = "density",
         add_cofactor: Union[Literal["x", "y", "both"], bool] = False,
         x_scale: Optional[Literal["biex", "log", "linear"]] = None,
         y_scale: Optional[Literal["biex", "log", "linear"]] = None,
         color_scale: Literal["biex", "log", "linear"] = "linear",
         cmap: Optional[str] = None,
         vmin: Optional[Union[float, int]] = None,
         vmax: Optional[Union[float, int]] = None,
         title: Optional[str] = None,
         figsize: tuple[float, float] = (4,4),
         return_dataframe: bool = False,
         return_fig: bool = False,
         ax: Optional[Axes] = None,
         show: bool = True,
         save: Optional[str] = None
         ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Plot for normal biaxial representation of cytometry data.

    Color the plot using annotations of observations (.obs) or expression of markers.
    Axes are customizable to `log`, `biex` or `linear` scale.

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
    sample_identifier
        Controls the data that are extracted. Defaults to None.
        If set, has to be one of the sample_IDs or the file_names.
    color
        The parameter that controls the coloring of the plot.
        Can be set to categorical variables from the .obs slot
        or continuous variables corresponding to channels.
        Default is set to 'density', which calculates the point
        density in the plot.
    add_cofactor
        if set, adds the cofactor as a line to the plot for visualization.
        if `x`, sets the cofactor for the x-axis,
        if `y`, sets the cofactor for the y-axis,
        if `both`, sets both axis cofactors
    x_scale
        sets the scale for the x axis. Has to be one of `biex`, `log`, `linear`.
        The value `biex` gets converted to `symlog` internally
    y_scale
        sets the scale for the y axis. Has to be one of `biex`, `log`, `linear`.
        The value `biex` gets converted to `symlog` internally
    color_scale
        sets the scale for the colorbar. Has to be one of `biex`, `log`, `linear`.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns `palette` and `cmap`
        parameters will use this value
    vmin
        minimum value to plot in the color vector
    vmax
        maximum value to plot in the color vector
    title
        sets the figure title. Optional
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    ax
        A :class:`~matplotlib.axes.Axes` created from matplotlib to plot into.
    show
        Whether to show the figure. Defaults to True.
    save
        Expects a file path including the file name.
        Saves the figure to the indicated path. Defaults to None.


    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes`
    If `return_fig==True` a :class:`~matplotlib.figure.Figure`
    If `return_dataframe==True` a :class:`~pandas.DataFrame` containing the data used for plotting


    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated'
    >>> fp.pl.biax(
    ...     dataset,
    ...     gate = "live",
    ...     layer = "compensated",
    ...     x_channel = "CD3",
    ...     y_channel = "SSC-A",
    ...     color = "batch",
    ...     x_scale = "biex",
    ...     y_scale = "linear"
    ... )
    
    """
    
    if x_scale not in ["biex", "linear", "log"] and x_scale is not None:
        raise ValueError("parameter x_scale has to be one of ['biex', 'linear', 'log']")
    if y_scale not in ["biex", "linear", "log"] and y_scale is not None:
        raise ValueError("parameter x_scale has to be one of ['biex', 'linear', 'log']")
    if color_scale not in ["biex", "linear", "log"] and color_scale is not None:
        raise ValueError("parameter x_scale has to be one of ['biex', 'linear', 'log']")
    
    adata = subset_gate(adata, gate = gate, as_view = True)
    if sample_identifier is not None:
        if _is_valid_sample_ID(adata, sample_identifier):
            adata = adata[adata.obs["sample_ID"] == str(sample_identifier),:]
        elif _is_valid_filename(adata, sample_identifier):
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
    else:
        transformed_color_vector = None
        color_vector = None

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

    if return_fig:
        return ax

    savefig_or_show(show = show, save = save)

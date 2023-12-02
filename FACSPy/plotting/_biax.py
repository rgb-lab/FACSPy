import warnings
import numpy as np
import pandas as pd

import matplotlib
from matplotlib.colors import ListedColormap, Normalize, SymLogNorm
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.figure import Figure

from anndata import AnnData

from typing import Union, Optional, Literal
from scipy.interpolate import interpn
from ._utils import (savefig_or_show,
                     _get_cofactor_from_var,
                     _color_var_is_categorical)
from .._utils import (subset_gate,
                      is_valid_filename,
                      is_valid_sample_ID,
                      _default_layer,
                      scatter_channels)
from ..exceptions._exceptions import CofactorNotFoundWarning

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

def _define_axis_scale(channel,
                       user_scale: Literal["biex", "linear", "log"]) -> bool:
    """
    decides if data are plotted on a linear or biex scale
    Log Scaling is not a default but has to be set by the user
    explicitly.
    """
    if user_scale:
        if user_scale == "biex":
            return "symlog"
        return user_scale
    if _is_scatter(channel):
        return "linear"
    return "symlog"

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

def _is_scatter(channel: str) -> bool:
    return any(k in channel for k in scatter_channels)

def _transform_data_to_scale(data: np.ndarray,
                             channel: str,
                             adata: AnnData,
                             user_scale) -> np.ndarray:
    scale = _define_axis_scale(channel, user_scale=user_scale)
    if scale == "linear":
        return data
    elif scale == "log":
        transformed = np.log10(data)
        # data can be negative to nan would be produced
        # which would mess up the density function
        transformed[np.where(np.isnan(transformed))] = 0.0
        return transformed
    elif scale == "symlog":
        cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                        channel)
        # for biex, we only log data above a certain value
        return np.arcsinh(data / cofactor)

def _retrieve_cofactor_or_set_to_default(adata, channel) -> float:
    try:
        return _get_cofactor_from_var(adata, channel)
    except KeyError:
        # which means cofactors were not calculated
        warnings.warn("Cofactor not found. Setting to 1000 for plotting",
                      CofactorNotFoundWarning)
        return 1000

#TODO: add legend
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
         cmap: str = None,
         vmin: float = None,
         vmax: float = None,
         figsize: tuple[float, float] = (4,4),
         title: Optional[str] = None,
         show: Optional[bool] = None,
         save: Optional[Union[str, bool]] = None,
         return_dataframe: bool = False,
         return_fig: bool = False) -> Optional[Figure]:
    
    if x_channel is None or y_channel is None:
        raise ValueError("Please provide x_channel and y_channel")
    if x_scale not in ["biex", "linear", "log"] and x_scale is not None:
        raise ValueError("parameter x_scale has to be one of ['biex', 'linear', 'log']")
    if y_scale not in ["biex", "linear", "log"] and y_scale is not None:
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

    if return_dataframe:
        return dataframe
    
    ### if plotting categorical colors with multiple sample
    ### this step is necessary to shuffle the colors
    dataframe = dataframe.sample(frac = 1)

    categorical_color = _color_var_is_categorical(dataframe[color])
    if not categorical_color:
        color_vector = dataframe[color].values.copy()
        if vmin:
            color_vector[np.where(color_vector < vmin)] = vmin
        if vmax:
            color_vector[np.where(color_vector > vmax)] = vmax

    continous_cmap = _get_cmap_biax(cmap, color)

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = figsize)
    plot_params = {
        "data": dataframe,
        "x": x_channel,
        "y": y_channel,
        "linewidth": 0,
        "s": 2,
        "hue": dataframe[color] if categorical_color else None,
        "palette": cmap or "Set1",
        "c": color_vector if not categorical_color else None,
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
    
    if color in adata.var_names:
        color_cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                              color)
    else:
        color_cofactor = 1

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
        custom_cmap = matplotlib.colormaps[continous_cmap]
        custom_colors = custom_cmap(np.linspace(0,1,256))
        if layer == "compensated":
            norm = SymLogNorm(vmin = np.min(color_vector),
                              vmax = np.max(color_vector),
                              linthresh = color_cofactor)
        else:
            norm = Normalize(vmin = np.min(color_vector),
                             vmax = np.max(color_vector))
        sm = plt.cm.ScalarMappable(cmap = ListedColormap(custom_colors),
                                   norm = norm)
        cbar = ax.figure.colorbar(sm,
                                  ax = ax)
        cbar.ax.set_ylabel(f"{layer} expression\n{color}",
                           rotation = 270,
                           labelpad = 30)
    plt.tight_layout()

    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)

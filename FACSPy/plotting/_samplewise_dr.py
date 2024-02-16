from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Literal, Union, Optional

from ._utils import (_get_uns_dataframe,
                     savefig_or_show,
                     _color_var_is_categorical,
                     _continous_color_vector,
                     _retrieve_cofactor_or_set_to_default,
                     _generate_continous_color_scale,
                     _transform_color_to_scale)

from .._utils import (_default_gate_and_default_layer,
                      _enable_gate_aliases,
                      reduction_names)

from .._settings import settings

def _samplewise_dr_plot(adata: AnnData,
                        gate: str,
                        data_metric: str,
                        data_group: str,
                        layer: str,
                        color: Optional[Union[str, list[str]]],
                        reduction: Literal["PCA", "MDS", "TSNE", "UMAP"],
                        color_scale: Literal["biex", "log", "linear"],
                        cmap: str = None,
                        vmin: float = None,
                        vmax: float = None,
                        figsize: tuple[float, float] = (4,3),
                        return_fig: bool = False,
                        return_dataframe = False,
                        ax: Axes = None,
                        save: bool = None,
                        show: bool = None):

    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{data_group}_{layer}")

    if return_dataframe:
        return data
    
    plotting_dimensions = _get_plotting_dimensions(reduction)
    categorical_color = _color_var_is_categorical(data[color])
    if not categorical_color:
        if color in adata.var_names and color_scale == "biex":
            color_cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                                  color)
        else:
            color_cofactor = None
        color_vector = _continous_color_vector(data,
                                               color,
                                               vmin,
                                               vmax)
        transformed_color_vector= _transform_color_to_scale(color_vector,
                                                            color_cofactor,
                                                            color_scale)

    continous_cmap = cmap or "viridis"

    plot_params = {
        "x": plotting_dimensions[0],
        "y": plotting_dimensions[1],
        "data": data,
        "hue": data[color] if categorical_color else None,
        "palette": cmap or settings.default_categorical_cmap,
        "c": transformed_color_vector if not categorical_color else None,
        "cmap": continous_cmap,
        "legend": "auto",
        "ax": ax
    }

    if ax is None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111)

    sns.scatterplot(**plot_params)
    
    if categorical_color:
        ax.legend(bbox_to_anchor = (1.1, 0.5), loc = "center left")
    else:
        cbar = _generate_continous_color_scale(color_vector = color_vector,
                                               cmap = continous_cmap,
                                               color_cofactor = color_cofactor,
                                               ax = ax,
                                               color_scale = color_scale)
        cbar.ax.set_ylabel(f"{layer} expression\n{color}",
                           rotation = 270,
                           labelpad = 30)

    ax.set_title(f"{reduction} samplewise reduction\ncolored by {color}")

    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)

    if show is False:
        return ax

def _get_plotting_dimensions(reduction: str):
    return reduction_names[reduction][:2]

@_default_gate_and_default_layer
@_enable_gate_aliases
def pca_samplewise(adata: AnnData,
                   gate: str = None,
                   layer: str = None,
                   color: str = None,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   color_scale: Literal["biex", "log", "linear"] = "linear",
                   cmap: str = None,
                   vmin: float = None,
                   vmax: float = None,
                   figsize: tuple[float, float] = (3,3),
                   save: bool = None,
                   show: bool = None,
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   ax: Axes = None,
                   ) -> Optional[Figure]:
    """
    Plot for visualizing sample-wise dimensionality reduction (PCA).

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
    color
        The parameter that controls the coloring of the plot.
        Can be set to categorical variables from the .obs slot
        or continuous variables corresponding to channels.
        Default is set to 'density', which calculates the point
        density in the plot.
    data_group
        Sets the groupby parameter that was used for samplewise dimred
        calculation. Using this value, the correct dataframe is extracted
        from adata.uns. Defaults to sample_ID
    data_metric
        Sets the data metric that was used for dimensionality reduction
        calculation. Can be one of 'mfi' or 'fop'
    color_scale
        Sets the scale for the colorbar. Has to be one of 'biex', 'log', 'linear'.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
    vmin
        Minimum value to plot in the color vector
    vmax
        Maximum value to plot in the color vector
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats
    save
        Expects a file path and a file name. saves the figure to the indicated path
    show
        Whether to show the figure
    return_dataframe
        If set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        If set to True, the figure is returned.
    """

    return _samplewise_dr_plot(reduction = "PCA",
                               adata = adata,
                               gate = gate,
                               layer = layer,
                               color = color,
                               data_metric = data_metric,
                               data_group = data_group,
                               color_scale = color_scale,
                               cmap = cmap,
                               vmin = vmin,
                               vmax = vmax,
                               ax = ax,
                               figsize = figsize,
                               return_dataframe = return_dataframe,
                               return_fig = return_fig,
                               save = save,
                               show = show)


@_default_gate_and_default_layer 
@_enable_gate_aliases
def mds_samplewise(adata: AnnData,
                   gate: str = None, 
                   layer: str = None,
                   color: str = None,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   color_scale: Literal["biex", "log", "linear"] = "linear",
                   cmap: str = None,
                   vmin: float = None,
                   vmax: float = None,
                   figsize: tuple[float, float] = (3,3),
                   ax: Axes = None,
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   save: bool = None,
                   show: bool = None
                   ) -> Optional[Figure]:
    """
    Plot for visualizing sample-wise dimensionality reduction (PCA).

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
    color
        The parameter that controls the coloring of the plot.
        Can be set to categorical variables from the .obs slot
        or continuous variables corresponding to channels.
        Default is set to 'density', which calculates the point
        density in the plot.
    data_group
        Sets the groupby parameter that was used for samplewise dimred
        calculation. Using this value, the correct dataframe is extracted
        from adata.uns. Defaults to sample_ID
    data_metric
        Sets the data metric that was used for dimensionality reduction
        calculation. Can be one of 'mfi' or 'fop'
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
    save
        expects a file path and a file name. saves the figure to the indicated path
    show
        whether to show the figure
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    """
 
    return _samplewise_dr_plot(reduction = "MDS",
                               adata = adata,
                               gate = gate,
                               layer = layer,
                               color = color,
                               data_metric = data_metric,
                               data_group = data_group,
                               color_scale = color_scale,
                               cmap = cmap,
                               vmin = vmin,
                               vmax = vmax,
                               figsize = figsize,
                               ax = ax,
                               return_dataframe = return_dataframe,
                               return_fig = return_fig,
                               save = save,
                               show = show)

@_default_gate_and_default_layer 
@_enable_gate_aliases
def umap_samplewise(adata: AnnData,
                    gate: str = None, 
                    layer: str = None,
                    color: str = None,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    color_scale: Literal["biex", "log", "linear"] = "linear",
                    cmap: str = None,
                    figsize: tuple[float, float] = (3,3),
                    vmin: float = None,
                    vmax: float = None,
                    save: bool = None,
                    show: bool = None,
                    return_dataframe: bool = False,
                    return_fig: bool = False,
                    ax: Axes = None,
                    ) -> Optional[Figure]:
    """
    Plot for visualizing sample-wise dimensionality reduction (UMAP).

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
    color
        The parameter that controls the coloring of the plot.
        Can be set to categorical variables from the .obs slot
        or continuous variables corresponding to channels.
        Default is set to 'density', which calculates the point
        density in the plot.
    data_group
        Sets the groupby parameter that was used for samplewise dimred
        calculation. Using this value, the correct dataframe is extracted
        from adata.uns. Defaults to sample_ID
    data_metric
        Sets the data metric that was used for dimensionality reduction
        calculation. Can be one of 'mfi' or 'fop'
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
    save
        expects a file path and a file name. saves the figure to the indicated path
    show
        whether to show the figure
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    """
    
    return _samplewise_dr_plot(reduction = "UMAP",
                               adata = adata,
                               gate = gate,
                               layer = layer,
                               color = color,
                               data_metric = data_metric,
                               data_group = data_group,
                               color_scale = color_scale,
                               cmap = cmap,
                               vmin = vmin,
                               vmax = vmax,
                               figsize = figsize,
                               ax = ax,
                               return_dataframe = return_dataframe,
                               return_fig = return_fig,
                               save = save,
                               show = show)


@_default_gate_and_default_layer 
@_enable_gate_aliases
def tsne_samplewise(adata: AnnData,
                    gate: str = None, 
                    layer: str = None,
                    color: str = None,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    color_scale: Literal["biex", "log", "linear"] = "linear",
                    cmap: str = None,
                    figsize: tuple[float, float] = (3,3),
                    vmin: float = None,
                    vmax: float = None,
                    save: bool = None,
                    show: bool = None,
                    return_dataframe: bool = False,
                    return_fig: bool = False,
                    ax: Axes = None
                    ) -> Optional[Figure]:
    """
    Plot for visualizing sample-wise dimensionality reduction (TSNE).

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
    color
        The parameter that controls the coloring of the plot.
        Can be set to categorical variables from the .obs slot
        or continuous variables corresponding to channels.
        Default is set to 'density', which calculates the point
        density in the plot.
    data_group
        Sets the groupby parameter that was used for samplewise dimred
        calculation. Using this value, the correct dataframe is extracted
        from adata.uns. Defaults to sample_ID
    data_metric
        Sets the data metric that was used for dimensionality reduction
        calculation. Can be one of 'mfi' or 'fop'
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
    save
        expects a file path and a file name. saves the figure to the indicated path
    show
        whether to show the figure
    return_dataframe
        if set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        if set to True, the figure is returned.
    """

    return _samplewise_dr_plot(reduction = "TSNE",
                               adata = adata,
                               gate = gate,
                               layer = layer,
                               color = color,
                               data_metric = data_metric,
                               data_group = data_group,
                               color_scale = color_scale,
                               cmap = cmap,
                               vmin = vmin,
                               vmax = vmax,
                               figsize = figsize,
                               ax = ax,
                               return_dataframe = return_dataframe,
                               return_fig = return_fig,
                               save = save,
                               show = show)

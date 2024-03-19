import warnings
import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Union, Optional, Literal

from ._utils import (savefig_or_show,
                     _retrieve_cofactor_or_set_to_default,
                     _generate_scale_kwargs,
                     _get_cofactor_from_var,
                     LINEPLOT_PARAMS)
from ._cofactor_plots import calculate_histogram_data

from .._utils import (_flatten_nested_list,
                      subset_gate,
                      _default_gate_and_default_layer,
                      _enable_gate_aliases)

def _map_pal_to_groupby(pal: dict,
                        data: pd.DataFrame,
                        groupby: str,
                        colorby: str) -> dict:
    """maps the original palette to new groupby variable by looking"""
    return {group: pal[data.loc[data[groupby] == group, colorby].iloc[0]] for group in data[groupby].unique()}

def _convert_to_mapping(dataframe: pd.DataFrame,
                        key_col: str,
                        value_col: str) -> dict:
    return {key_value: dataframe.loc[dataframe[key_col] == key_value, value_col].iloc[0]
            for key_value in dataframe[key_col].unique()}

def _append_metadata_obs(adata: AnnData,
                         expression_data: pd.DataFrame) -> pd.DataFrame:
    expression_data[adata.obs.columns] = adata.obs
    return expression_data

def _convert_expression_to_histogram_data(expression_data: pd.DataFrame,
                                          marker: str,
                                          groupby: str) -> pd.DataFrame:
    group_values = list(expression_data[groupby].unique())
    histogram_df = pd.DataFrame(
        data = {groupby: _flatten_nested_list([[group for _ in range (100)] for group in group_values])},
        columns = [groupby, "x", "y"],
        index = range(100 * len(group_values))
    )
    
    for group in  group_values:
        group_spec_expression_data = expression_data.loc[expression_data[groupby] == group, [groupby, marker]]
        x, y = calculate_histogram_data(group_spec_expression_data,
                                        {"x": marker})
        histogram_df.loc[histogram_df[groupby] == group, ["x", "y"]] = np.vstack([x, y]).T

    return histogram_df

def _append_colorby_variable(adata: AnnData,
                             dataframe: pd.DataFrame,
                             colorby: str,
                             groupby: str) -> pd.DataFrame:
    mapping = _convert_to_mapping(adata.uns["metadata"].to_df(),
                                  key_col = groupby,
                                  value_col = colorby)
    dataframe[colorby] = dataframe[groupby].map(mapping)
    return dataframe

#TODO: check if mapping is possible: either the colorby is not in metadata or the grouping by sampleID leads to multiple outputs, not mappable.
@_default_gate_and_default_layer
@_enable_gate_aliases
def marker_density(adata: AnnData,
                   gate: str = None,
                   layer: str = None,
                   marker: str = None,
                   groupby: str = "sample_ID",
                   colorby: str = None,
                   highlight: Optional[Union[str, list[str]]] = None,
                   ridge: bool = False,
                   add_cofactor: bool = False,
                   cmap: str = "Set1",
                   plot_height: float = 1,
                   plot_spacing: float = -0.5,
                   plot_aspect: float = 1,
                   linewidth: float = 0.5,
                   xlim: Optional[tuple[float, float]] = None,
                   x_scale: Literal["biex", "log", "linear"] = "linear",
                   figsize: tuple[float, float] = (3,3),
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   ax: Optional[Axes] = None,
                   show: bool = True,
                   save: Optional[str] = None
                   ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
    Histogram Plot of marker expression. Either as a Ridge-Plot or as 
    a overlayed lineplot.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    marker
        The channel to be displayed. Has to be in adata.var_names.
    groupby
        Controls the x axis and the grouping of the data points.
    colorby
        Controls the coloring of the individual histogram lines.
    highlight
        Controls whether to highlight a specific group of `colorby`.
        Highlighted groups will be red, other groups will be gray.
    ridge
        If `True`, a ridge-plot is shown. If `False`, an overlayed
        line plot is shown.
    add_cofactor
        If True, adds a line at the level of the cofactor. 
    cmap
        Sets the colormap for plotting. Defaults to `Set1`
    plot_height
        Height of the plot if `ridge==True`. For `ridge==False` use
        the `figsize` parameter to control the plot dimensions.
    plot_spacing
        Spacing of the plots if `ridge==True`. Negative numbers will
        lead to overlapping of the histogram plots. Defaults to -0.5.
    plot_aspect
        Controls the aspect of the plot if `ridge==True`.
    linewidth
        Controls the linewidth of the histogram lines. Ignored if `ridge==True`.
    xlim
        Sets the limits of the x-axis.
    x_scale
        sets the scale for the x axis. Has to be one of `biex`, `log`, `linear`.
        The value `biex` gets converted to `symlog` internally
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
        Only if `ridge==False`.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    ax
        A :class:`~matplotlib.axes.Axes` created from matplotlib to plot into.
        Only works if `ridge==False`.
    show
        Whether to show the figure. Defaults to True.
    save
        Expects a file path including the file name.
        Saves the figure to the indicated path. Defaults to None.

    Returns
    -------
    If `show==False` and `ridge==False` a :class:`~matplotlib.axes.Axes`
    If `show==False` and `ridge==True` a :class:`~matplotlib.figure.Figure`
    If `return_fig==True` a :class:`~matplotlib.figure.Figure`
    If `return_dataframe==True` a :class:`~pandas.DataFrame` containing the data used for plotting

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.pl.marker_density(
    ...     dataset,
    ...     gate = "live",
    ...     layer = "compensated",
    ...     marker = "CD3",
    ...     groupby = "condition",
    ...     colorby = "sex"
    ... )

    
    """

    if not isinstance(highlight, list) and highlight is not None:
        highlight = [highlight]

    adata = subset_gate(adata,
                        gate = gate,
                        as_view = True)

    expression_data = adata.to_df(layer = layer)
    expression_data = _append_metadata_obs(adata, expression_data)

    histogram_df = _convert_expression_to_histogram_data(expression_data = expression_data,
                                                         marker = marker,
                                                         groupby = groupby)
    if colorby is not None:
        histogram_df = _append_colorby_variable(adata = adata,
                                                dataframe = histogram_df,
                                                colorby = colorby,
                                                groupby = groupby)
        histogram_df[colorby] = histogram_df[colorby].astype(str)
    histogram_df[groupby] = histogram_df[groupby].astype(str)
    
    if return_dataframe:
        return histogram_df
    
    if highlight is not None:
        colorby_pal = {
            group: "red" if group in highlight else "grey"
            for group in list(histogram_df[colorby].unique())
        }
    elif colorby is not None:
        user_defined_cmap = sns.color_palette(
            cmap,
            len(histogram_df[colorby].unique())
        )
        colorby_pal = {group: user_defined_cmap[i]
            for i, group in enumerate(list(histogram_df[colorby].unique()))
        }
    else:
        colorby_pal = None

    x_channel_cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                              marker)
    x_scale_kwargs = _generate_scale_kwargs(marker,
                                            x_scale,
                                            x_channel_cofactor)

    if ridge:
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        if groupby != colorby and colorby_pal is not None:
            pal = _map_pal_to_groupby(colorby_pal, histogram_df, groupby, colorby)
        else:
            pal = colorby_pal

        fig: sns.FacetGrid = sns.FacetGrid(histogram_df,
                                           row = groupby,
                                           hue = groupby,
                                           aspect = plot_aspect,
                                           despine = True,
                                           height = plot_height,
                                           palette = pal)
        
        fig.map(sns.lineplot, "x", "y", clip_on = False)

        if add_cofactor is True:
            fig.map(plt.axvline, x = x_channel_cofactor, color = "black")
        
        for _ax, _ax_name in zip(fig.axes.flat, fig._axes_dict.keys()):
            _ax: Axes
            _ax.fill_between(x = histogram_df.loc[histogram_df[groupby] == _ax, "x"].to_numpy(dtype = np.float32),
                             y1 = histogram_df.loc[histogram_df[groupby] == _ax, "y"].to_numpy(dtype = np.float32),
                             y2 = 0,
                             alpha = 0.1)
            _ax.set_xscale(**x_scale_kwargs)
            _ax.set_title("")
            _ax.set_ylabel("")
                          
            if groupby != colorby:
                _ax.text(0.9, .2, _ax_name,
                         fontsize = 10,
                         color="black",
                         transform = _ax.transAxes)
            _ax.set_xlabel(f'{marker}\n{layer}\nexpression', fontsize = 10)
            if xlim is not None:
                _ax.set_xlim(xlim)

        if colorby is not None:
            handles = [Patch(facecolor = colorby_pal[name]) for name in colorby_pal]
            labels = list(colorby_pal.keys())
            sns.reset_orig()
            group_legend = plt.legend(handles,
                                      labels,
                                      loc = "center left",
                                      title = colorby,
                                      bbox_to_anchor = (1, 0.5),
                                      bbox_transform = fig.figure.transFigure)
        
            fig.figure.add_artist(group_legend)

        fig.figure.subplots_adjust(hspace = plot_spacing)
        
        # TODO: REMOVE uncomment if in bad mood :)
        #print("Nico war hier möhöhöhö")
        
    else:
        if ax is None:
            fig: Figure = plt.figure(figsize = figsize)
            ax = fig.add_subplot(111)
        else:
            fig = None

        sns.lineplot(data = histogram_df,
                     x = "x",
                     y = "y",
                     hue = colorby,
                     style = groupby,
                     palette = colorby_pal,
                     linewidth = linewidth,
                     ax = ax,
                     **LINEPLOT_PARAMS)
        
        ax.set_xscale(**x_scale_kwargs)

        if add_cofactor:
            ax.axvline(x = x_channel_cofactor, color = "black")

        if colorby is not None:
            handles, labels = ax.get_legend_handles_labels()
            colorby_labels = np.array(
                [[handle, label]
                 for handle, label in zip(handles, labels)
                 if any(k in label for k in [colorby] + list(histogram_df[colorby].unique()))]
            )
            ax.legend(colorby_labels[:,0],
                      colorby_labels[:,1],
                      bbox_to_anchor = (1.05, 0.5),
                      loc = "center left",
                      title = colorby)
        else:
            ax.legend().remove()
        ax.set_title(f"Marker expression {marker}\nper sample ID")
        ax.set_ylabel("Density (norm)")
        ax.set_xlabel(f"{layer} expression")
        if xlim is not None:
            ax.set_xlim(xlim)


    if return_fig:
        return fig

    savefig_or_show(show = show, save = save)

    if show is False:
        return ax if not ridge else fig
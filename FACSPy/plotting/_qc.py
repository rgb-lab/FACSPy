from anndata import AnnData
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Union, Optional, Literal

from ._categorical_stripplot import _categorical_strip_box_plot
from ._utils import savefig_or_show

from .._utils import (subset_gate,
                      convert_gate_to_obs,
                      _find_gate_path_of_gate,
                      _find_parent_gate,
                      _find_grandparent_gate,
                      _find_parent_population,
                      _find_grandparent_population,
                      _find_current_population,
                      _default_gate,
                      GATE_SEPARATOR)
from ..exceptions._exceptions import AnalysisNotPerformedError, HierarchyError
from .._settings import settings

def _find_y_label(adata: AnnData,
                  freq_of: Optional[Union[str, list[str], Literal["parent", "grandparent"]]],
                  gate: Union[str, list[str]]):
    
    if freq_of == "parent":
        return _find_parent_population(_find_gate_path_of_gate(adata, gate))
    if freq_of == "grandparent":
        return _find_grandparent_population(_find_gate_path_of_gate(adata, gate))
    return "All Cells" if freq_of in ["root", "all"] else freq_of

def _prepare_dataframe_gate_frequency(adata: AnnData,
                                      gate: Union[str, list[str]],
                                      freq_of: Optional[Union[str, list[str]]],
                                      groupby: Optional[Union[str, list[str]]],
                                      splitby: Optional[str]) -> pd.DataFrame:
    
    if "gate_frequencies" not in adata.uns:
        raise AnalysisNotPerformedError("fp.tl.gate_frequencies")
    df = adata.uns["gate_frequencies"].reset_index()
    
    gate = _find_gate_path_of_gate(adata, gate)

    if freq_of not in gate.split(GATE_SEPARATOR)[:-1] and freq_of not in [None, "parent", "grandparent", "all"]:
        raise HierarchyError
    
    if freq_of == "parent":
        freq_of = _find_parent_gate(gate)
    elif freq_of is None or freq_of == "all":
        freq_of = "root"
    elif freq_of == "grandparent":
        freq_of = _find_grandparent_gate(gate)
    elif GATE_SEPARATOR not in freq_of:
        freq_of = _find_gate_path_of_gate(adata, freq_of)

    df = df.loc[(df["gate"] == gate) & (df["freq_of"] == freq_of)]
    
    obs = _prepare_dataframe_cell_counts(adata, groupby, splitby = splitby)
    obs = obs.drop("counts", axis = 1)
    return obs.merge(df, on = "sample_ID")

def _prepare_dataframe_cell_counts(adata: AnnData,
                                   groupby: Optional[str],
                                   splitby: Optional[str]):
    if groupby == "sample_ID":
        groupings = [groupby]
    elif groupby is None:
        groupings = ["sample_ID"]
    else:
        groupings = ["sample_ID", groupby]
    #groupings = ["sample_ID"] + copy.copy(groupby) if "sample_ID" not in groupby else copy.copy(groupby)
    if splitby is not None:
        groupings.append(splitby)
    groupings = list(set(groupings)) ## in case the user chooses groupby and splitby as the same
    #if groupby == None:
    #    return adata.obs["sample_ID"].value_counts().to_frame(name = "counts").reset_index(names = "sample_ID")

    return adata.obs[groupings].value_counts().to_frame(name = "counts").reset_index()

@_default_gate
def gate_frequency(adata: AnnData,
                   gate: Union[str, list[str]] = None,
                   freq_of: Optional[Union[str, list[str], Literal["parent", "grandparent", "all"]]] = None,
                   groupby: Optional[str] = None,
                   splitby: Optional[str] = None,
                   cmap: str = None,
                   stat_test: str = "Kruskal",
                   order: list[str] = None,
                   figsize: tuple[float, float] = (3,3),
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   ax: Axes = None,
                   save: bool = None,
                   show: bool = None):
    """
    Plots the gate frequency in comparison to a defined gate.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    freq_of
        Sets the reference gate of which the frequency of the chosen gate
        is displayed
    groupby
        controls the x axis and the grouping of the data points
    splitby
        controls the coloring of the data points. Defaults to None.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
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
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots

    Returns
    -------

    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """

    data = _prepare_dataframe_gate_frequency(adata,
                                             gate,
                                             freq_of,
                                             groupby,
                                             splitby)

    if return_dataframe:
        return data

    plot_params = {
        "data": data,
        "x": groupby,
        "y": "freq",
        "hue": splitby,
        "palette": cmap or settings.default_categorical_cmap if splitby else None,
        "order": order
    }

    fig, ax = _categorical_strip_box_plot(ax = ax,
                                          data = data,
                                          plot_params = plot_params,
                                          groupby = groupby,
                                          splitby = splitby,
                                          stat_test = stat_test,
                                          figsize = figsize)

    ax.set_title(f"gate frequency per {groupby}\ngate: {_find_current_population(gate)}")
    ax.set_xlabel("")
    ax.set_ylabel(f"freq. of\n{_find_y_label(adata, freq_of, gate)}")

    if return_fig:
        return fig    

    savefig_or_show(show = show, save = save)

    if show is False:
        return ax


@_default_gate
def cell_counts(adata: AnnData,
                gate: str = None,
                groupby: Optional[Union[str, list[str]]] = None,
                splitby: Optional[str] = None,
                cmap: str = None,
                order: list[str] = None,
                stat_test: Optional[str] = "Kruskal",
                figsize: tuple[float, float] = (3,3),
                return_dataframe: bool = False,
                return_fig: bool = False,
                ax: Axes = None,
                save: bool = None,
                show: bool = None) -> Optional[Union[Figure, Axes]]:
    """
    Plots the cell counts of a specific population.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    groupby
        controls the x axis and the grouping of the data points
    colorby
        controls the coloring of the data points. Defaults to None.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
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
    ax
        Optional parameter. Sets user defined ax from for example plt.subplots

    Returns
    -------

    if `show==False` a :class:`~matplotlib.axes.Axes`
    
    """

    if gate is not None:
        if gate not in adata.obs.columns:
            convert_gate_to_obs(adata, gate)
        adata = subset_gate(adata,
                            gate = gate,
                            copy = False,
                            as_view = True)

    data = _prepare_dataframe_cell_counts(adata,
                                          groupby,
                                          splitby)
    
    if return_dataframe:
        return data
    
    plot_params = {
        "data": data,
        "x": groupby,
        "y": "counts",
        "hue": splitby,
        "palette": cmap or settings.default_categorical_cmap if splitby else None,
        "order": order

    }

    fig, ax = _categorical_strip_box_plot(ax = ax,
                                          data = data,
                                          plot_params = plot_params,
                                          groupby = groupby,
                                          splitby = splitby,
                                          stat_test = stat_test,
                                          figsize = figsize)

    ax.set_title(f"cell counts\ngrouped by {groupby}")
    ax.set_xlabel("")
    ax.set_ylabel("cell counts")

    if return_fig:
        return fig    

    savefig_or_show(show = show, save = save)

    if show is False:
        return ax
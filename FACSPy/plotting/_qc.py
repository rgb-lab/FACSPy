from anndata import AnnData
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Union, Optional, Literal

from ._categorical_stripplot import _categorical_strip_box_plot
from ._utils import savefig_or_show

from .._utils import (subset_gate,
                      _find_gate_path_of_gate,
                      _find_parent_gate,
                      _find_grandparent_gate,
                      _find_parent_population,
                      _find_grandparent_population,
                      _find_current_population,
                      _default_gate,
                      _enable_gate_aliases,
                      _is_parent)
from ..exceptions._exceptions import AnalysisNotPerformedError, HierarchyError
from .._settings import settings

def _find_y_label(adata: AnnData,
                  freq_of: Union[str, Literal["parent", "grandparent", "all"]],
                  gate: str):
    
    if freq_of == "parent":
        return _find_parent_population(_find_gate_path_of_gate(adata, gate))
    if freq_of == "grandparent":
        return _find_grandparent_population(_find_gate_path_of_gate(adata, gate))
    return "All Cells" if freq_of in ["root", "all"] else freq_of

def _prepare_dataframe_gate_frequency(adata: AnnData,
                                      gate: str,
                                      freq_of: Union[str, Literal["parent", "grandparent", "all"]],
                                      groupby: str,
                                      splitby: Optional[str]) -> pd.DataFrame:
    
    if "gate_frequencies" not in adata.uns:
        raise AnalysisNotPerformedError("fp.tl.gate_frequencies")
    df = adata.uns["gate_frequencies"].reset_index()
    
    gate = _find_gate_path_of_gate(adata, gate)

    if freq_of == "parent":
        freq_of = _find_parent_gate(gate)
    elif freq_of == "grandparent":
        freq_of = _find_grandparent_gate(gate)
    elif freq_of is None or freq_of == "all":
        freq_of = "root"
    else: 
        if not _is_parent(adata, gate, freq_of):
            raise HierarchyError
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
    
    if splitby is not None:
        groupings.append(splitby)
    groupings = list(set(groupings)) ## in case the user chooses groupby and splitby as the same

    return adata.obs[groupings].value_counts().to_frame(name = "counts").reset_index()

@_default_gate
@_enable_gate_aliases
def gate_frequency(adata: AnnData,
                   gate: str,
                   freq_of: Union[str, Literal["parent", "grandparent", "all"]],
                   groupby: str,
                   splitby: Optional[str] = None,
                   cmap: Optional[str] = None,
                   order: Optional[Union[list[str], str]] = None,
                   stat_test: Optional[str] = "Kruskal",
                   figsize: tuple[float, float] = (3,3),
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   ax: Optional[Axes] = None,
                   show: bool = True,
                   save: Optional[str] = None
                   ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
    """\
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
    order
        specifies the order of x-values.
    stat_test
        Statistical test that is used for the p-value calculation. One of
        `Kruskal` and `Wilcoxon`. Defaults to Kruskal.
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
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.tl.gate_frequencies(dataset)
    >>> fp.pl.gate_frequency(
    ...     dataset,
    ...     gate = "live",
    ...     groupby = "condition",
    ...     splitby = "sex"
    ... )
    
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
@_enable_gate_aliases
def cell_counts(adata: AnnData,
                gate: str,
                groupby: str, 
                splitby: Optional[str] = None,
                cmap: Optional[str] = None,
                order: Optional[Union[list[str], str]] = None,
                stat_test: Optional[str] = "Kruskal",
                figsize: tuple[float, float] = (3,3),
                return_dataframe: bool = False,
                return_fig: bool = False,
                ax: Optional[Axes] = None,
                show: bool = True,
                save: Optional[str] = None
                ) -> Optional[Union[Figure, Axes, pd.DataFrame]]:
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
    splitby
        The parameter controlling additional split along the groupby-axis.
    cmap
        Sets the colormap for plotting. Can be continuous or categorical, depending
        on the input data. When set, both seaborns 'palette' and 'cmap'
        parameters will use this value
    order
        specifies the order of x-values.
    stat_test
        Statistical test that is used for the p-value calculation. One of
        `Kruskal` and `Wilcoxon`. Defaults to Kruskal.
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
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.pl.cell_counts(
    ...     dataset,
    ...     gate = "live",
    ...     groupby = "condition",
    ...     splitby = "sex"
    ... )
    
    """

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



from anndata import AnnData
from typing import Union, Optional, Literal

import pandas as pd
from matplotlib import pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..exceptions._exceptions import HierarchyError

from ._utils import savefig_or_show
from ._baseplot import (stripboxplot,
                        barplot,
                        label_plot_basic,
                        adjust_legend)
from ._basestats import add_statistic

from .._utils import (GATE_SEPARATOR,
                      find_gate_path_of_gate,
                      find_parent_gate,
                      find_grandparent_gate,
                      subset_gate,
                      find_parent_population,
                      find_grandparent_population,
                      find_current_population,
                      convert_gate_to_obs,
                      _default_gate)
from ..exceptions._exceptions import AnalysisNotPerformedError

def _find_y_label(adata: AnnData,
                  freq_of: Optional[Union[str, list[str], Literal["parent", "grandparent"]]],
                  gate: Union[str, list[str]]):
    
    if freq_of == "parent":
        return find_parent_population(find_gate_path_of_gate(adata, gate))
    if freq_of == "grandparent":
        return find_grandparent_population(find_gate_path_of_gate(adata, gate))
    return "All Cells" if freq_of in ["root", "all"] else freq_of

def _prepare_dataframe_gate_frequency(adata: AnnData,
                                      gate: Union[str, list[str]],
                                      freq_of: Optional[Union[str, list[str]]],
                                      groupby: Optional[Union[str, list[str]]],
                                      colorby: Optional[str]) -> pd.DataFrame:
    
    if "gate_frequencies" not in adata.uns:
        raise AnalysisNotPerformedError("fp.tl.gate_frequencies")
    df = adata.uns["gate_frequencies"].reset_index()
    
    if GATE_SEPARATOR not in gate:
        gate = find_gate_path_of_gate(adata, gate)

    if freq_of not in gate.split(GATE_SEPARATOR)[:-1] and freq_of not in [None, "parent", "grandparent", "all"]:
        raise HierarchyError
    
    if freq_of == "parent":
        freq_of = find_parent_gate(gate)
    elif freq_of is None or freq_of == "all":
        freq_of = "root"
    elif freq_of == "grandparent":
        freq_of = find_grandparent_gate(gate)
    elif GATE_SEPARATOR not in freq_of:
        freq_of = find_gate_path_of_gate(adata, freq_of)

    df = df.loc[(df["gate"] == gate) & (df["freq_of"] == freq_of)]
    
    obs = _prepare_dataframe_cell_counts(adata, groupby, colorby = colorby)
    obs = obs.drop("counts", axis = 1)
    return obs.merge(df, on = "sample_ID")

def _prepare_dataframe_cell_counts(adata: AnnData,
                                   groupby: Optional[str],
                                   colorby: Optional[str]):
    import copy
    groupings = ["sample_ID"] + copy.copy(groupby) if "sample_ID" not in groupby else copy.copy(groupby)
    if colorby != [None]:
        groupings += colorby
    groupings = list(set(groupings)) ## in case the user chooses groupby and colorby as the same
    if groupby == [None]:
        return adata.obs["sample_ID"].value_counts().to_frame(name = "counts").reset_index(names = "sample_ID")

    return adata.obs[groupings].value_counts().to_frame(name = "counts").reset_index()

@_default_gate
def gate_frequency(adata: AnnData,
                   gate: Union[str, list[str]] = None,
                   groupby: Optional[str] = None,
                   colorby: Optional[str] = None,
                   freq_of: Optional[Union[str, list[str], Literal["parent", "grandparent", "all"]]] = None,
                   figsize: tuple[float, float] = (4,3),
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   save: bool = None,
                   show: bool = None):
    
    if not isinstance(groupby, list):
        groupby = [groupby]

    if not isinstance(colorby, list):
        colorby = [colorby]

    df = _prepare_dataframe_gate_frequency(adata,
                                           gate,
                                           freq_of,
                                           groupby,
                                           colorby)

    if return_dataframe:
        return df

    ncols = 1
    nrows = 1
    figsize = figsize

    plot_params = {
        "x": groupby[0],
        "y": "freq",
        "hue": colorby[0],
        "data": df
    }

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
    if groupby == ["sample_ID"]:
        ax = barplot(ax,
                     plot_params = plot_params)

    else:
        ax = stripboxplot(ax,
                          plot_params = plot_params)
        try:
            ax = add_statistic(ax = ax,
                                test = "Kruskal",
                                dataframe = df,
                                groupby = groupby[0],
                                plot_params = plot_params)
        except ValueError as e:
            if str(e) != "All numbers are identical in kruskal":
                raise ValueError from e
            else:
                print("warning... Values were uniform, no statistics to plot.")

    ax = label_plot_basic(ax = ax,
                          title = f"gate frequency per {groupby[0]}\ngate: {find_current_population(gate)}",
                          y_label = f"freq. of\n{_find_y_label(adata, freq_of, gate)}",
                          x_label = "")

    if colorby != [None]:
        ax = adjust_legend(ax)
    else:
        ax.legend().remove()
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")

    if return_fig:
        return fig    
    plt.tight_layout()
    savefig_or_show(show = show, save = save)


@_default_gate
def cell_counts(adata: AnnData,
                gate: str = None,
                groupby: Optional[Union[str, list[str]]] = None,
                colorby: Optional[str] = None,
                figsize: tuple[float, float] = (4,3),
                return_dataframe: bool = False,
                return_fig: bool = False,
                save: bool = None,
                show: bool = None) -> Optional[Union[Figure, Axes]]:

    if gate is not None:
        if gate not in adata.obs.columns:
            convert_gate_to_obs(adata, gate)
        adata = subset_gate(adata,
                            gate = gate,
                            copy = False,
                            as_view = True)

    if not isinstance(groupby, list):
        groupby = [groupby]
    
    if not isinstance(colorby, list):
        colorby = [colorby]
    
    df = _prepare_dataframe_cell_counts(adata,
                                        groupby,
                                        colorby)
    
    if return_dataframe:
        return df 
    
    ncols = 1
    nrows = 1
    figsize = figsize

    plot_params = {
        "data": df,
        "x": groupby[0],
        "y": "counts",
        "hue": colorby[0],

    }

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
    if groupby == ["sample_ID"]:
        ax = barplot(ax,
                     plot_params = plot_params)

    else:
        ax = stripboxplot(ax,
                          plot_params = plot_params)
        try:
            ax = add_statistic(ax = ax,
                                test = "Kruskal",
                                dataframe = df,
                                groupby = groupby[0],
                                plot_params = plot_params)
        except ValueError as e:
            if str(e) != "All numbers are identical in kruskal":
                raise ValueError from e
            else:
                print("warning... Values were uniform, no statistics to plot.")

    ax = label_plot_basic(ax = ax,
                          title = f"cell counts\ngrouped by {groupby[0]}",
                          y_label = "cell counts",
                          x_label = "")
    
    if colorby != [None]:
        ax = adjust_legend(ax)
    else:
        ax.legend().remove()

    
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation = 45,
                       ha = "center")
        
    if return_fig:
        return fig
    
    plt.tight_layout()
    savefig_or_show(save = save, show = show)
        

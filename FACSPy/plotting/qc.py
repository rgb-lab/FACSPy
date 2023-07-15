

from anndata import AnnData
from typing import Union, Optional, Literal

import pandas as pd
from matplotlib import pyplot as plt

from matplotlib.axes import Axes

from ..exceptions.exceptions import HierarchyError

from .utils import create_boxplot, savefig_or_show

from ..utils import (GATE_SEPARATOR,
                     find_gate_path_of_gate,
                     find_parent_gate,
                     find_grandparent_gate,
                     subset_gate,
                     find_parent_population,
                     find_grandparent_population)

def prepare_dataframe_cell_counts(adata: AnnData,
                      groupby: Optional[Union[str, list[str]]]):
    
    groupings = ["sample_ID"] + groupby if "sample_ID" not in groupby else groupby
    if groupby == [None]:
        return adata.obs["sample_ID"].value_counts().to_frame(name = "counts").reset_index(names = "sample_ID")

    return adata.obs[groupings].value_counts().to_frame(name = "counts").reset_index()


def prepare_dataframe_gate_frequency(adata: AnnData,
                                     gate: Union[str, list[str]],
                                     freq_of: Optional[Union[str, list[str]]],
                                     groupby: Optional[Union[str, list[str]]]) -> pd.DataFrame:
    
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
    
    obs = prepare_dataframe_cell_counts(adata, groupby)
    obs = obs.drop("counts", axis = 1)
    return obs.merge(df, on = "sample_ID")


def gate_frequency(adata: AnnData,
                   gate: Union[str, list[str]],
                   freq_of: Optional[Union[str, list[str], Literal["parent", "grandparent", "all"]]] = None,
                   groupby: Optional[Union[str, list[str]]] = None,
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   save: bool = None,
                   show: bool = None):
    
    if not isinstance(groupby, list):
        groupby = [groupby]

    df = prepare_dataframe_gate_frequency(adata,
                                          gate,
                                          freq_of,
                                          groupby)

    if return_dataframe:
        return df

    ncols = 1
    nrows = len(groupby)
    figsize = (len(df) / 10, 3 * len(groupby)) 
    
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)

    for i, grouping in enumerate(groupby):
        plot_params = {
            "x": "sample_ID" if grouping is None else grouping,
            "y": "freq",
            "data": df
        }

        if len(groupby) > 1:
            ax[i] = create_boxplot(ax[i],
                                   grouping,
                                   plot_params)
            ax[i] = label_frequency_plot(ax[i],
                                         grouping,
                                         gate,
                                         find_y_label(adata, freq_of, gate)
                                         )
        else:
            ax = create_boxplot(ax,
                                grouping,
                                plot_params)
            ax = label_frequency_plot(ax,
                                      grouping,
                                      gate,
                                      find_y_label(adata, freq_of, gate)
                                      )
    if return_fig:
        return fig    
    plt.tight_layout()
    savefig_or_show(show = show, save = save)

def label_cell_count_plot(ax: Axes,
                          grouping: str,
                          population: str) -> Axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")
    ax.set_title(f"cell counts per {grouping or 'sample_ID'}\npopulation: {population or 'All cells'}")
    ax.set_ylabel("counts per sample")
    ax.set_xlabel("")
    return ax

def find_y_label(adata: AnnData,
                 freq_of: Optional[Union[str, list[str], Literal["parent", "grandparent"]]],
                 gate: Union[str, list[str]]):
    
    if freq_of == "parent":
        return find_parent_population(find_gate_path_of_gate(adata, gate))
    if freq_of == "grandparent":
        return find_grandparent_population(find_gate_path_of_gate(adata, gate))
    return "All Cells" if freq_of in ["root", "all"] else freq_of

def label_frequency_plot(ax: Axes,
                         grouping: str,
                         gate: str,
                         freq_of: str) -> Axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")
    ax.set_title(f"gate frequency per {grouping}\ngate: {gate}")
    ax.set_ylabel(f"freq. of\n{freq_of}")
    ax.set_xlabel("")
    return ax

def cell_counts(adata: AnnData,
                groupby: Optional[Union[str, list[str]]] = None,
                population: Optional[str] = None,
                return_dataframe: bool = False,
                return_fig: bool = False,
                save: bool = None,
                show: bool = None):

    if population is not None:
        adata = subset_gate(adata, gate = population, copy = False, as_view = True)

    if not isinstance(groupby, list):
        groupby = [groupby]
    
    df = prepare_dataframe_cell_counts(adata, groupby)
    
    if return_dataframe:
        return df 
    
    ncols = 1,
    nrows = len(groupby)
    figsize = (len(df) / 10, 3 * len(groupby)) 
    
    fig, ax = plt.subplots(ncols = 1, nrows = nrows, figsize = figsize)

    
    for i, grouping in enumerate(groupby):
        plot_params = {
            "x": "sample_ID" if grouping is None else grouping,
            "y": "counts",
            "data": df
        }
        
        if len(groupby) > 1:
            ax[i] = create_boxplot(ax[i],
                                   grouping,
                                   plot_params)
            ax[i] = label_cell_count_plot(ax = ax[i],
                                          grouping = grouping,
                                          population = population)
        else:
            ax = create_boxplot(ax,
                                grouping,
                                plot_params)
            ax = label_cell_count_plot(ax = ax,
                                       grouping = grouping,
                                       population = population)
        
    if return_fig:
        return fig
    
    plt.tight_layout()
    savefig_or_show(save = save, show = show)
        

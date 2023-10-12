from anndata import AnnData
import pandas as pd
from typing import Union, Literal, Optional

from matplotlib.axes import Axes

from matplotlib import pyplot as plt

from ._basestats import add_statistic
from ._baseplot import adjust_legend

from ._basestats import add_statistic
from ._baseplot import barplot, stripboxplot, label_plot_basic
from .utils import (get_uns_dataframe,
                    savefig_or_show)

def fop(adata: AnnData,
        marker: Union[str, list[str]],
        groupby: Union[str, list[str]] = None,
        colorby: Optional[str] = None,
        order: list[str] = None,
        gate: str = None,
        overview: bool = False,
        data_group: Optional[Union[str, list[str]]] = "sample_ID",
        data_metric: Literal["mfi", "fop", "gate_frequency"] = "fop",
        data_origin: Literal["compensated", "transformed"] = "compensated",
        figsize: tuple[float, float] = (3,3),
        return_dataframe: bool = False,
        return_fig: bool = False,
        save: bool = None,
        show: bool = None):
    
    data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = f"{data_metric}_{data_group}_{data_origin}")

    if return_dataframe:
        return data
    
    mfi_fop_baseplot(adata = adata,
                     dataframe = data,
                     marker = marker,
                     groupby = groupby,
                     colorby = colorby,
                     gate = gate,
                     assay = "fop",
                     overview = overview,
                     figsize = figsize,
                     order = order,
                     return_fig = return_fig,
                     save = save,
                     show = show)

def mfi(adata: AnnData,
        gate: str,
        marker: Union[str, list[str]],
        colorby: Optional[str] = None,
        order: list[str] = None,
        groupby: Union[str, list[str]] = None,
        data_group: Optional[Union[str, list[str]]] = "sample_ID",
        data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
        data_origin: Literal["compensated", "transformed"] = "compensated",
        overview: bool = False,
        figsize: tuple[float, float] = (3,3),
        return_dataframe: bool = False,
        return_fig: bool = False,
        save: bool = None,
        show: bool = None):

    data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = f"{data_metric}_{data_group}_{data_origin}")

    if return_dataframe:
        return data
    
    mfi_fop_baseplot(adata = adata,
                     dataframe = data,
                     marker = marker,
                     groupby = groupby,
                     colorby = colorby,
                     gate = gate,
                     assay = "mfi",
                     overview = overview,
                     figsize = figsize,
                     order = order,
                     return_fig = return_fig,
                     save = save,
                     show = show)

def mfi_fop_baseplot(adata: AnnData,
                     dataframe: pd.DataFrame,
                     marker: Union[str, list[str]],
                     groupby: Union[str, list[str]],
                     colorby: str,
                     assay: Literal["mfi", "fop"],
                     order: list[str] = None,
                     gate: str = None,
                     overview: bool = False,
                     figsize: tuple[float, float] = None,
                     return_fig: bool = False,
                     save: bool = None,
                     show: bool = None):
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    if overview:
        if marker:
            print("warning... marker argument is ignored when using overview")
        marker = adata.var_names.to_list()

    if not isinstance(marker, list):
        marker = [marker]

    if not isinstance(groupby, list):
        groupby = [groupby]

    if not isinstance(colorby, list):
        colorby = [colorby]

    ncols = 1
    nrows = len(groupby)
    figsize = figsize
    plot_params = {
        "data": dataframe,
        "x": groupby[0],
        "y": marker[0],
        "hue": colorby[0],
        "hue_order": order if colorby[0] is not None else None,
        "order": order if colorby[0] is None else None
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
                                dataframe = dataframe,
                                groupby = groupby[0],
                                plot_params = plot_params)
        except ValueError as e:
            if str(e) != "All numbers are identical in kruskal":
                raise ValueError from e
            else:
                print("warning... Values were uniform, no statistics to plot.")

    ax = label_plot_basic(ax = ax,
                          title = f"{marker[0]}\ngrouped by {groupby[0]}",
                          y_label = f"{marker[0]}",
                          x_label = "")
    
    ax = adjust_legend(ax,
                       title = colorby[0] or None)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")
        
    if return_fig:
        return fig

    plt.tight_layout()
    savefig_or_show(save = save, show = show)

def label_plot(ax: Axes,
               marker: str,
               grouping: str,
               assay: Literal["mfi", "fop"]) -> Axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")
    ax.set_title(f"{marker} expression\nper {grouping}")
    ax.set_ylabel("expression" if assay == "mfi" else "fraction positive")
    ax.set_xlabel("")
    return ax
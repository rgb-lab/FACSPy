from anndata import AnnData
import pandas as pd
import numpy as np
from typing import Union, Literal, Optional

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib import pyplot as plt
import seaborn as sns


from statannotations.Annotator import Annotator

from ..utils import (find_gate_path_of_gate,
                     create_comparisons)

from ..exceptions.exceptions import AnalysisNotPerformedError

from .utils import (create_boxplot,
                    turn_off_missing_plots,
                    get_uns_dataframe)
### mfi plot



def fop(adata: AnnData,
        marker: Union[str, list[str]],
        groupby: Union[str, list[str]] = None,
        colorby: Optional[str] = None,
        order: list[str] = None,
        gate: str = None,
        overview: bool = False,
        return_dataframe: bool = False):
    
    data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = "fop",
                             column_identifier_name = "sample_ID")

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
                     order = order)

def mfi(adata: AnnData,
        marker: Union[str, list[str]],
        colorby: Optional[str] = None,
        order: list[str] = None,
        groupby: Union[str, list[str]] = None,
        gate: str = None,
        overview: bool = False,
        return_dataframe: bool = False):

    data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = "mfi",
                             column_identifier_name = "sample_ID")

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
                     order = order)

def mfi_fop_baseplot(adata: AnnData,
                     dataframe: pd.DataFrame,
                     marker: Union[str, list[str]],
                     groupby: Union[str, list[str]],
                     colorby: str,
                     assay: Literal["mfi", "fop"],
                     order: list[str] = None,
                     gate: str = None,
                     overview: bool = False):
    
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

    for grouping in groupby:
        ncols = 4 if overview else 1
        nrows = int(np.ceil(len(marker) / 4)) if overview else len(marker)
        figsize = (12 if overview
                   else 3 if colorby is None else 4,
                   int(np.ceil(len(marker) / 4)) * 3 if overview
                   else 4 * len(marker))
        
        fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
        
        if len(marker) > 1:
            ax = ax.flatten()
        
        for i, _marker in enumerate(marker):
            print(f"plotting statistics for {_marker}")
            if grouping is None:
                print("... warning: You are computing statistics for each sample... this takes a while")
            plot_params = {
                "x": "sample_ID" if grouping is None else grouping,
                "y": _marker,
                "data": dataframe,
                "order": order,
                "hue": colorby
            }
            if len(marker)>1:
                ax[i] = create_boxplot(ax = ax[i],
                                       grouping = grouping,
                                       plot_params = plot_params)
                ax[i] = label_plot(ax = ax[i],
                                   marker = _marker,
                                   grouping = grouping,
                                   assay = assay)
                if colorby is not None:
                    ax[i] = adjust_legend(ax[i])
                try:
                    ax[i] = add_statistic(ax = ax[i],
                                          test = "Kruskal",
                                          dataframe = dataframe,
                                          groupby = "sample_ID" if grouping is None else grouping,
                                          plot_params = plot_params)
                except ValueError as e:
                    if str(e) != "All numbers are identical in kruskal":
                        raise ValueError from e
                    else:
                        print("warning... Values were uniform, no statistics to plot.")
            else:
                ax = create_boxplot(ax = ax,
                                    grouping = grouping,
                                    plot_params = plot_params)
                ax = label_plot(ax = ax,
                                marker = _marker,
                                grouping = grouping,
                                assay = assay)
                if colorby is not None:
                    ax = adjust_legend(ax)
                try:
                    ax = add_statistic(ax = ax,
                                       test = "Kruskal",
                                       dataframe = dataframe,
                                       groupby = "sample_ID" if grouping is None else grouping,
                                       plot_params = plot_params)
                except ValueError as e:
                    if str(e) != "All numbers are identical in kruskal":
                        raise ValueError from e
                    else:
                        print("warning... Values were uniform, no statistics to plot.")

                
        if len(marker) > 1:
            ax = turn_off_missing_plots(ax)
        ax = np.reshape(ax, (ncols, nrows))

        plt.tight_layout()
        plt.show()

def adjust_legend(ax: Axes) -> Axes:
    ax.legend(loc = "upper left",
              bbox_to_anchor = (1.05, 0.5))
    return ax

def add_statistic(ax: Axes,
                  test: str,
                  dataframe: pd.DataFrame,
                  groupby: str,
                  plot_params) -> Axes:
    
    pairs = create_comparisons(dataframe, groupby)
    annotator = Annotator(ax,
                          pairs,
                          **plot_params,
                          verbose = False)
    annotator.configure(test = test, text_format = "star", loc = "inside")
    annotator.apply_and_annotate()

    return ax
    
def label_plot(ax: Axes,
               marker: str,
               grouping: str,
               assay: Literal["mfi", "fop"]) -> Axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")
    ax.set_title(f"{marker} expression\nper {grouping}")
    ax.set_ylabel("expression" if assay == "mfi" else "fraction positive")
    ax.set_xlabel("")
    return ax
from anndata import AnnData
import pandas as pd
import numpy as np
from typing import Union, Literal

from matplotlib.figure import Figure
from matplotlib.axis import Axis

from matplotlib import pyplot as plt
import seaborn as sns

from ..utils import find_gate_path_of_gate

from ..exceptions.exceptions import AnalysisNotPerformedError

from .utils import create_boxplot, append_metadata, turn_off_missing_plots, prep_uns_dataframe
### mfi plot



def fop(adata: AnnData,
        markers: Union[str, list[str]],
        groupby: Union[str, list[str]],
        order: list[str] = None,
        gate: str = None,
        overview: bool = False):
    
    try:
        fop_data = adata.uns["fop"]
        fop_data = prep_uns_dataframe(adata, fop_data)
    except KeyError as e:
        raise AnalysisNotPerformedError("fop") from e
    
    mfi_fop_baseplot(adata = adata,
                     dataframe = fop_data,
                     markers = markers,
                     groupby = groupby,
                     gate = gate,
                     assay = "fop",
                     overview = overview,
                     order = order)

def mfi(adata: AnnData,
        markers: Union[str, list[str]],
        order: list[str] = None,
        groupby: Union[str, list[str]] = None,
        gate: str = None,
        overview: bool = False):

    try:
        mfi_data = adata.uns["mfi"]
        mfi_data = prep_uns_dataframe(adata, mfi_data)
    except KeyError as e:
        raise AnalysisNotPerformedError("mfi") from e

    mfi_fop_baseplot(adata = adata,
                     dataframe = mfi_data,
                     markers = markers,
                     groupby = groupby,
                     gate = gate,
                     assay = "mfi",
                     overview = overview,
                     order = order)


from itertools import combinations
def create_comparisons(data: pd.DataFrame,
                       groupby: str) -> list[tuple[str, str]]:
    return list(combinations(data[groupby].unique(), 2))

from statannotations.Annotator import Annotator
def add_statistic(ax: Axis,
                  test: str,
                  dataframe: pd.DataFrame,
                  groupby: str,
                  marker: str,
                  plot_params) -> Axis:
    
    pairs = create_comparisons(dataframe, groupby)

    annotator = Annotator(ax,
                          pairs,
                          **plot_params,
                          verbose = False)
    annotator.configure(test = test, text_format = "star", loc = "inside")
    annotator.apply_and_annotate()

    return ax
    
    

def mfi_fop_baseplot(adata: AnnData,
                     dataframe: pd.DataFrame,
                     markers: Union[str, list[str]],
                     groupby: Union[str, list[str]],
                     assay: Literal["mfi", "fop"],
                     order: list[str] = None,
                     gate: str = None,
                     overview: bool = False):
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    if overview:
        if markers:
            print("warning... markers argument is ignored when using overview")
        markers = adata.var_names.to_list()
    
    if not isinstance(markers, list):
        markers = [markers]
    if not isinstance(groupby, list):
        groupby = [groupby]

    for grouping in groupby:
        ncols = 4 if overview else 1
        nrows = int(np.ceil(len(markers) / 4)) if overview else len(markers)
        figsize = (12 if overview
                   else 3,
                   int(np.ceil(len(markers) / 4)) * 3 if overview
                   else 5* len(markers)) 
        fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
        if len(markers) > 1:
            ax = ax.flatten()
        for i, marker in enumerate(markers):
            print(marker)
            plot_params = {
                "x": "sample_ID" if grouping is None else grouping,
                "y": marker,
                "data": gate_specific_mfis,
                "order": order
            }
            if len(markers)>1:
                ax[i] = create_boxplot(ax = ax[i],
                                    grouping = grouping,
                                    plot_params = plot_params)
                ax[i] = label_plot(ax = ax[i],
                                   marker = marker,
                                   grouping = grouping,
                                   assay = assay)
                ax[i] = add_statistic(ax = ax[i],
                                      test = "Kruskal",
                                      dataframe = gate_specific_mfis,
                                      groupby = "sample_ID" if grouping is None else grouping,
                                      marker = marker,
                                      plot_params = plot_params)
            else:
                ax = create_boxplot(ax = ax,
                                    grouping = grouping,
                                    plot_params = plot_params)
                ax = label_plot(ax = ax,
                                marker = marker,
                                grouping = grouping,
                                assay = assay)
                ax = add_statistic(ax = ax,
                                   test = "Kruskal",
                                   dataframe = gate_specific_mfis,
                                   groupby = "sample_ID" if grouping is None else grouping,
                                   marker = marker,
                                   plot_params = plot_params)
        if len(markers) > 1:
            ax = turn_off_missing_plots(ax)
        ax = np.reshape(ax, (ncols, nrows))
        
        plt.tight_layout()
        plt.show()



def label_plot(ax: Axis,
               marker: str,
               grouping: str,
               assay: Literal["mfi", "fop"]) -> Axis:
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = "center")
    ax.set_title(f"{marker} expression\nper {grouping}")
    ax.set_ylabel("expression" if assay == "mfi" else "fraction positive")
    ax.set_xlabel("")
    return ax
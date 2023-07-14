import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure

from typing import Literal, Union, Optional

from .utils import get_uns_dataframe, turn_off_missing_plots, savefig_or_show
from ..utils import find_gate_path_of_gate, reduction_names

from ..exceptions.exceptions import AnalysisNotPerformedError


def samplewise_dr_plot(adata: AnnData,
                       dataframe: pd.DataFrame,
                       groupby: Optional[Union[str, list[str]]],
                       reduction: Literal["PCA", "MDS", "TSNE", "UMAP"],
                       overview: bool = False,
                       return_fig: bool = False,
                       save: bool = None,
                       show: bool = None):
    
    if not isinstance(groupby, list):
        groupby = [groupby]
    
    if overview:
        if groupby:
            print("warning... groupby argument are ignored when using overview")
        #TODO add fluo_columns as negative selection!
        groupby = [
            entry
            for entry in dataframe.columns.to_list()
            if all(
                k not in entry
                for k in [
                    "sample_ID",
                    "date",
                    "file_name",
                    "UMAP",
                    "TSNE",
                    "gate_path",
                    "PCA",
                    "MDS",
                ]
            )
        ]

    plotting_dimensions = get_plotting_dimensions(reduction)

    ncols = 4 if overview else 1
    nrows = int(np.ceil(len(groupby) / 4)) if overview else len(groupby)
    figsize = (12 if overview else 3,
               int(np.ceil(len(groupby) / 4)) * 3.2 if overview else 3.2 * len(groupby)) 

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
    if len(groupby) > 1:
        ax = ax.flatten()
    for i, grouping in enumerate(groupby):
        plot_params = {
            "x": plotting_dimensions[0],
            "y": plotting_dimensions[1],
            "data": dataframe,
            "hue": grouping
                   if dataframe[grouping].dtype.__class__.__name__ == "CategoricalDtype"
                   else None,
            "c": dataframe[grouping]
                 if dataframe[grouping].dtype.__class__.__name__ != "CategoricalDtype" 
                 else None
        }

        if len(groupby) > 1:
            ax[i] = create_scatterplot(ax = ax[i],
                                       plot_params = plot_params)
            ax[i].set_title(f"{reduction} samplewise reduction\ngroupbyed by {grouping}")
            #sns.move_legend(ax[i], "center right")
            if dataframe[grouping].dtype.__class__.__name__ == "CategoricalDtype":
                handles, labels = ax[i].get_legend_handles_labels()
                ax[i].legend(handles,
                             labels,
                             loc = "upper left",
                             bbox_to_anchor = (1,1),
                             title = grouping
                             )

        else:
            ax = create_scatterplot(ax = ax,
                                    plot_params = plot_params)
            ax.set_title(f"{reduction} samplewise reduction\ngroupbyed by {grouping}")
            #sns.move_legend(ax, "center right", bbox_to_anchor = (2,0.5))
            if dataframe[grouping].dtype.__class__.__name__ == "CategoricalDtype":
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,
                          labels,
                          loc = "upper left",
                          bbox_to_anchor = (1,1),
                          title = grouping
                          )

    if len(groupby) > 1:
        ax = turn_off_missing_plots(ax)
    ax = np.reshape(ax, (ncols, nrows))

    if return_fig:
        return fig
    savefig_or_show(show = show, save = save)
            

def get_plotting_dimensions(reduction: str):
    return reduction_names[reduction][:2]

def create_scatterplot(ax: Axis,
                       plot_params: dict) -> Axis:
    return sns.scatterplot(**plot_params,
                           edgecolor = "black",
                           ax = ax)

def pca_samplewise(adata: AnnData,
                   groupby: str,
                   gate: str, 
                   on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   overview: bool = False,
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   save: bool = None,
                   show: bool = None
                   ) -> Optional[Figure]:
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = on,
                             column_identifier_name = "sample_ID")

    if return_dataframe:
        return data
    
    samplewise_dr_plot(adata = adata,
                       dataframe = data,
                       reduction = "PCA",
                       groupby = groupby,
                       overview = overview,
                       return_fig = return_fig,
                       save = save,
                       show = show)
    
def mds_samplewise(adata: AnnData,
                   groupby: str,
                   gate: str, 
                   on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   overview: bool = False,
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   save: bool = None,
                   show: bool = None
                   ) -> Optional[Figure]:
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = on,
                             column_identifier_name = "sample_ID")

    if return_dataframe:
        return data
    
    samplewise_dr_plot(adata = adata,
                       dataframe = data,
                       reduction = "MDS",
                       groupby = groupby,
                       overview = overview,
                       return_fig = return_fig,
                       save = save,
                       show = show)
import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure

from typing import Literal, Union, Optional

from .utils import prep_uns_dataframe, turn_off_missing_plots
from ..utils import find_gate_path_of_gate, reduction_names

from ..exceptions.exceptions import AnalysisNotPerformedError


def samplewise_dr_plot(adata: AnnData,
                       dataframe: pd.DataFrame,
                       color: Optional[Union[str, list[str]]],
                       reduction: Literal["PCA", "MDS", "TSNE", "UMAP"],
                       gate: Optional[str] = None,
                       overview: bool = False):
    if not isinstance(color, list):
        color = [color]
    if gate is None:
        raise TypeError("A Gate has to be provided")
    if overview:
        if color:
            print("warning... color argument are ignored when using overview")
        color = [
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

    full_gate_path = find_gate_path_of_gate(adata, gate)
    gate_specific_mfis = dataframe.loc[dataframe["gate_path"] == full_gate_path, :]

    plotting_dimensions = get_plotting_dimensions(reduction)

    ncols = 4 if overview else 1
    nrows = int(np.ceil(len(color) / 4)) if overview else len(color)
    figsize = (12 if overview else 3,
               int(np.ceil(len(color) / 4)) * 3.2 if overview else 3.2 * len(color)) 

    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = figsize)
    if len(color) > 1:
        ax = ax.flatten()
    for i, grouping in enumerate(color):
        plot_params = {
            "x": plotting_dimensions[0],
            "y": plotting_dimensions[1],
            "data": gate_specific_mfis,
            "hue": grouping
                   if gate_specific_mfis[grouping].dtype.__class__.__name__ == "CategoricalDtype"
                   else None,
            "c": gate_specific_mfis[grouping]
                 if gate_specific_mfis[grouping].dtype.__class__.__name__ != "CategoricalDtype" 
                 else None
        }

        if len(color) > 1:
            ax[i] = create_scatterplot(ax = ax[i],
                                       plot_params = plot_params)
            ax[i].set_title(f"{reduction} samplewise reduction\ncolored by {grouping}")

        else:
            ax = create_scatterplot(ax = ax,
                                    plot_params = plot_params)
            ax.set_title(f"{reduction} samplewise reduction\ncolored by {grouping}")

    if len(color) > 1:
        ax = turn_off_missing_plots(ax)
    ax = np.reshape(ax, (ncols, nrows))

    plt.tight_layout()
    plt.show()
            

def get_plotting_dimensions(reduction: str):
    return reduction_names[reduction][:2]

def create_scatterplot(ax: Axis,
                       plot_params: dict) -> Axis:
    return sns.scatterplot(**plot_params,
                           edgecolor = "black",
                           ax = ax)

def pca_samplewise(adata: AnnData,
                   color: str,
                   gate: str, 
                   on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   overview: bool = False
                   ) -> Optional[Figure]:
    
    try:
        data = adata.uns[on]
        data = prep_uns_dataframe(adata, data)
        #TODO
        #_ = data["PCA1"]
    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e
    
    samplewise_dr_plot(adata = adata,
                       dataframe = data,
                       reduction = "PCA",
                       color = color,
                       gate = gate,
                       overview = overview)
    
def mds_samplewise(adata: AnnData,
                   color: str,
                   gate: str, 
                   on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   overview: bool = False
                   ) -> Optional[Figure]:
    
    try:
        data = adata.uns[on]
        data = prep_uns_dataframe(adata, data)
    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e
    
    samplewise_dr_plot(adata = adata,
                       dataframe = data,
                       reduction = "MDS",
                       color = color,
                       gate = gate,
                       overview = overview)
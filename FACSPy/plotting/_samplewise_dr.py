import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure

from typing import Literal, Union, Optional

from ._utils import _get_uns_dataframe, turn_off_missing_plots, savefig_or_show
from .._utils import _default_gate_and_default_layer, reduction_names

def _samplewise_dr_plot(adata: AnnData,
                        dataframe: pd.DataFrame,
                        groupby: Optional[Union[str, list[str]]],
                        reduction: Literal["PCA", "MDS", "TSNE", "UMAP"],
                        palette: str = None,
                        cmap: str = None,
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

    plotting_dimensions = _get_plotting_dimensions(reduction)

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
                   if isinstance(dataframe[grouping].dtype, pd.CategoricalDtype)
                   else None,
            "c": dataframe[grouping]
                 if not isinstance(dataframe[grouping].dtype, pd.CategoricalDtype)
                 else None,
            "palette": palette or "tab10",
            "cmap": cmap or "viridis"
        }

        if len(groupby) > 1:
            ax[i] = create_scatterplot(ax = ax[i],
                                       plot_params = plot_params)
            ax[i].set_title(f"{reduction} samplewise reduction\ngrouped by {grouping}")
            if isinstance(dataframe[grouping].dtype, pd.CategoricalDtype):
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
            ax.set_title(f"{reduction} samplewise reduction\ngrouped by {grouping}")
            if isinstance(dataframe[grouping].dtype, pd.CategoricalDtype):
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
            

def _get_plotting_dimensions(reduction: str):
    return reduction_names[reduction][:2]

def create_scatterplot(ax: Axis,
                       plot_params: dict) -> Axis:
    return sns.scatterplot(**plot_params,
                           edgecolor = "black",
                           ax = ax)

@_default_gate_and_default_layer
def pca_samplewise(adata: AnnData,
                   groupby: str,
                   gate: str = None,
                   layer: str = None,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   palette: str = None,
                   cmap: str = None,
                   overview: bool = False,
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   save: bool = None,
                   show: bool = None
                   ) -> Optional[Figure]:
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{data_group}_{layer}")

    if return_dataframe:
        return data
    
    _samplewise_dr_plot(adata = adata,
                        dataframe = data,
                        reduction = "PCA",
                        groupby = groupby,
                        palette = palette,
                        cmap = cmap,
                        overview = overview,
                        return_fig = return_fig,
                        save = save,
                        show = show)

@_default_gate_and_default_layer 
def mds_samplewise(adata: AnnData,
                   groupby: str,
                   gate: str = None, 
                   layer: str = None,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   palette: str = None,
                   cmap: str = None,
                   overview: bool = False,
                   return_dataframe: bool = False,
                   return_fig: bool = False,
                   save: bool = None,
                   show: bool = None
                   ) -> Optional[Figure]:
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{data_group}_{layer}")

    if return_dataframe:
        return data
    
    _samplewise_dr_plot(adata = adata,
                        dataframe = data,
                        reduction = "MDS",
                        groupby = groupby,
                        palette = palette,
                        cmap = cmap,
                        overview = overview,
                        return_fig = return_fig,
                        save = save,
                        show = show)

@_default_gate_and_default_layer 
def umap_samplewise(adata: AnnData,
                    groupby: str,
                    gate: str = None, 
                    layer: str = None,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    palette: str = None,
                    cmap: str = None,
                    overview: bool = False,
                    return_dataframe: bool = False,
                    return_fig: bool = False,
                    save: bool = None,
                    show: bool = None
                    ) -> Optional[Figure]:
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{data_group}_{layer}")

    if return_dataframe:
        return data
    
    _samplewise_dr_plot(adata = adata,
                        dataframe = data,
                        reduction = "UMAP",
                        groupby = groupby,
                        palette = palette,
                        cmap = cmap,
                        overview = overview,
                        return_fig = return_fig,
                        save = save,
                        show = show)

@_default_gate_and_default_layer 
def tsne_samplewise(adata: AnnData,
                    groupby: str,
                    gate: str = None, 
                    layer: str = None,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    palette: str = None,
                    cmap: str = None,
                    overview: bool = False,
                    return_dataframe: bool = False,
                    return_fig: bool = False,
                    save: bool = None,
                    show: bool = None
                    ) -> Optional[Figure]:
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{data_group}_{layer}")

    if return_dataframe:
        return data
    
    _samplewise_dr_plot(adata = adata,
                        dataframe = data,
                        reduction = "TSNE",
                        groupby = groupby,
                        palette = palette,
                        cmap = cmap,
                        overview = overview,
                        return_fig = return_fig,
                        save = save,
                        show = show)
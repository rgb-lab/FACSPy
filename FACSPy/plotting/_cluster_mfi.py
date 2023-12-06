from anndata import AnnData
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Literal, Optional, Union

from ._utils import (_scale_data,
                     _calculate_sample_distance,
                     _calculate_linkage,
                     _get_uns_dataframe,
                     _scale_cbar_to_heatmap,
                     _calculate_correlation_data,
                     _remove_dendrogram,
                     add_annotation_plot,
                     savefig_or_show)
from ._clustermap import create_clustermap
from ._frequency_plots import _prep_dataframe_cluster_freq

from .._utils import _default_gate_and_default_layer

def cluster_mfi(): return None

def prepare_plot_data(adata: AnnData,
                      raw_data: pd.DataFrame,
                      scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]],
                      copy: bool = False
                      ) -> pd.DataFrame:
    plot_data = raw_data.copy() if copy else raw_data
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    if scaling is not None:
        plot_data[fluo_columns] = _scale_data(plot_data[fluo_columns], scaling)
    return plot_data

@_default_gate_and_default_layer
def cluster_heatmap(adata: AnnData,
                    gate: str = None,
                    layer: str = None,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = "MinMaxScaler",
                    corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                    cluster_method: Literal["correlation", "distance"] = "distance",
                    annotate: Optional[Union[Literal["frequency"], str]] = None,
                    annotation_kwargs: dict = {},
                    cmap: str = "RdYlBu_r",
                    figsize: Optional[tuple[float, float]] = (5,3.8),
                    y_label_fontsize: Optional[Union[int, float]] = 10,
                    return_dataframe: bool = False,
                    return_fig: bool = False,
                    save: bool = None,
                    show: bool = None) -> Optional[Figure]:
    """
    Plots a heatmap where every column corresponds to one cluster and the
    rows display the marker expression.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where Rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler`(Z-score)
    corr_method
        correlation method that is used for hierarchical clustering by cluster correlation.
        if cluster_method == `distance`, this parameter is ignored. One of `pearson`, `spearman` 
        or `kendall`.
    cluster_method
        Method for hierarchical clustering of displayed clusters. If `correlation`, the correlation
        specified by corr_method is computed (default: pearson). If `distance`, the euclidean
        distance is computed.
    annotate
        Parameter to control the annotation plot. Default: `frequency`. Adds a plot on top of
        the heatmap to display cluster-specific data. Other valid values are marker names as
        contained in adata.var_names
    annotation_kwargs
        Used to specify and customize the annotation plot. 
    y_label_fontsize
        controls the fontsize of the marker labels
    cmap
        Sets the colormap for plotting the markers
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats
    save
        Expects a file path and a file name. saves the figure to the indicated path
    show
        Whether to show the figure
    return_dataframe
        If set to True, returns the raw data that are used for plotting. vmin and vmax
        are not set.
    return_fig
        If set to True, the figure is returned.

    Returns
    -------

    if `show==False` a :class:`~seaborn.ClusterGrid`
 
    """
    
    raw_data = _get_uns_dataframe(adata = adata,
                                  gate = gate,
                                  table_identifier = f"{data_metric}_{data_group}_{layer}")
    
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    plot_data = prepare_plot_data(adata = adata,
                                  raw_data = raw_data,
                                  copy = True,
                                  scaling = scaling)
    plot_data = plot_data.set_index(data_group)
    if return_dataframe:
        return plot_data

    if cluster_method == "correlation":
        col_linkage = _calculate_linkage(_calculate_correlation_data(plot_data[fluo_columns].T, corr_method))

    elif cluster_method == "distance":
        col_linkage = _calculate_linkage(_calculate_sample_distance(plot_data[fluo_columns]))

    clustermap = create_clustermap(data = plot_data[fluo_columns].T,
                                   row_cluster = True,
                                   col_linkage = col_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   cbar_kws = {"label": "scaled expression" if scaling else "expression",
                                               "orientation": 'vertical'},
                                   vmin = None,
                                   vmax = None
                                   )
    
    indices = [t.get_text() for t in np.array(clustermap.ax_heatmap.get_xticklabels())]
    
    heatmap = clustermap.ax_heatmap
    _scale_cbar_to_heatmap(clustermap = clustermap,
                           heatmap_position = heatmap.get_position(),
                           cbar_padding = 1.05,
                           loc = "right")
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 45, ha = "center")
    heatmap.yaxis.set_ticks_position("left")
    heatmap.set_yticklabels(heatmap.get_yticklabels(),
                            fontsize = y_label_fontsize)
    heatmap.set_ylabel("")
    _remove_dendrogram(clustermap, which = "y")
    heatmap.set_xlabel("cluster")

    if annotate is not None:
        if annotate == "frequency":
            annot_frame = _prep_dataframe_cluster_freq(
                adata,
                groupby = annotation_kwargs.get("groupby", "sample_ID"),
                cluster_key = annotation_kwargs.get("cluster_key", "leiden"),
                normalize = annotation_kwargs.get("normalize", True),
            )
        elif annotate in adata.var_names:
            raw_data = raw_data.set_index(data_group)
            annot_frame = raw_data[annotate]

        add_annotation_plot(adata = adata,
                            annotate = annotate,
                            annot_frame = annot_frame,
                            indices = indices,
                            clustermap = clustermap,
                            y_label_fontsize = y_label_fontsize,
                            y_label = annotate
                            )
    if return_fig:
        return clustermap

    savefig_or_show(show = show, save = save)
    
    if show is False:
        return clustermap







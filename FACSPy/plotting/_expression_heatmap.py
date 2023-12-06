from anndata import AnnData
import numpy as np
import pandas as pd

from matplotlib.figure import Figure

from typing import Literal, Optional, Union

from ._utils import (_scale_data,
                     _map_obs_to_cmap,
                     _calculate_sample_distance,
                     _calculate_linkage,
                     _remove_dendrogram,
                     _add_metaclusters,
                     _remove_ticklabels,
                     _remove_ticks,
                     _scale_cbar_to_heatmap,
                     _add_categorical_legend_to_clustermap,
                     _calculate_correlation_data,
                     add_annotation_plot,
                     _get_uns_dataframe,
                     ANNOTATION_CMAPS,
                     savefig_or_show)
from ._clustermap import create_clustermap

from .._utils import _default_gate_and_default_layer

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
def expression_heatmap(adata: AnnData,
                       gate: str = None,
                       layer: str = None,
                       annotate: Optional[Union[str, list[str]]] = None,
                       plot_annotate: Optional[str] = None,
                       data_group: Optional[Union[str, list[str]]] = "sample_ID",
                       data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cluster_method: Literal["correlation", "distance"] = "distance",
                       cmap: str = "RdBu_r",
                       metaclusters: Optional[int] = None,
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc",
                       figsize: Optional[tuple[int, int]] = (5,3.8),
                       y_label_fontsize: Optional[Union[int, float]] = 10,
                       return_dataframe: bool = False,
                       return_fig: bool = False,
                       save: bool = None,
                       show: bool = None) -> Optional[Figure]:
    """
    Plot for expression heatmap. Rows are the individual channels and columns are the data points.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    annotate
        controls the annotated variables on top of the plot.
    plot_annotate
        creates a second plot on top of the heatmap where marker expressions can
        be shown. 
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
        correlation method that is used for hierarchical clustering by sample correlation.
        if cluster_method == `distance`, this parameter is ignored. One of `pearson`, `spearman` 
        or `kendall`.
    cluster_method
        Method for hierarchical clustering of displayed samples. If `correlation`, the correlation
        specified by corr_method is computed (default: pearson). If `distance`, the euclidean
        distance is computed.
    metaclusters
        controls the n of metaclusters to be computed
    label_metaclusters_in_dataset
        Whether to label the calculated metaclusters and write into the metadata
    label_metaclusters_key
        Column name that is used to store the metaclusters in
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

    if not isinstance(annotate, list):
        annotate = [annotate]    
    
    raw_data = _get_uns_dataframe(adata = adata,
                                  gate = gate,
                                  table_identifier = f"{data_metric}_{data_group}_{layer}")
    ### QUICK FIX FOR MISSING SAMPLES! CHECK CHECK CHECK!
    raw_data = raw_data.dropna(axis = 0, how = "any")
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    plot_data = prepare_plot_data(adata = adata,
                                  raw_data = raw_data,
                                  copy = True,
                                  scaling = scaling)
    ### QUICK FIX FOR MISSING SAMPLES! CHECK CHECK CHECK!
    plot_data = plot_data.dropna(axis = 0, how = "any")
    
    if cluster_method == "correlation":
        col_linkage = _calculate_linkage(_calculate_correlation_data(plot_data[fluo_columns].T, corr_method))
    
    elif cluster_method == "distance":
        col_linkage = _calculate_linkage(_calculate_sample_distance(plot_data[fluo_columns]))

    if metaclusters is not None:
        annotate += ["metacluster"]
        plot_data = _add_metaclusters(adata = adata,
                                      data = plot_data,
                                      row_linkage = col_linkage,
                                      n_clusters = metaclusters,
                                      sample_IDs = raw_data["sample_ID"],
                                      label_metaclusters = label_metaclusters_in_dataset,
                                      label_metaclusters_key = label_metaclusters_key)

    plot_data = plot_data.set_index(data_group)
    if return_dataframe:
        return plot_data
    ### for the heatmap, the dataframe is transposed so that sample_IDs are the columns
    clustermap = create_clustermap(data = plot_data[fluo_columns].T,
                                   col_colors = [
                                       _map_obs_to_cmap(plot_data, group, ANNOTATION_CMAPS[i])
                                       for i, group in enumerate(annotate)
                                   ],
                                   row_cluster = True,
                                   col_linkage = col_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   cbar_kws = {"label": "scaled expression", "orientation": 'horizontal'},
                                   vmin = 0 if scaling == "MinMaxScaler" else None,
                                   vmax = 1 if scaling == "MinMaxScaler" else None
                                   )

    indices = [t.get_text() for t in np.array(clustermap.ax_heatmap.get_xticklabels())]
    ax = clustermap.ax_heatmap
    _scale_cbar_to_heatmap(clustermap,
                           heatmap_position = ax.get_position(),
                           cbar_padding = 0.5)    

    _remove_ticklabels(ax, which = "x")
    _remove_ticks(ax, which = "x")
    _remove_dendrogram(clustermap, which = "y")
    ax.set_xlabel("")

    ax.yaxis.set_ticks_position("left")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = y_label_fontsize)
    clustermap.ax_row_dendrogram.set_visible(False)

    _add_categorical_legend_to_clustermap(clustermap,
                                          heatmap = ax,
                                          data = plot_data,
                                          annotate = annotate)

    if plot_annotate is not None:
        if plot_annotate in adata.var_names:
            raw_data = raw_data.set_index(data_group)
            annot_frame = raw_data[plot_annotate]

        add_annotation_plot(adata = adata,
                            annotate = plot_annotate,
                            annot_frame = annot_frame,
                            indices = indices,
                            clustermap = clustermap,
                            y_label_fontsize = y_label_fontsize,
                            y_label = data_metric)

    if return_fig:
        return clustermap
    
    savefig_or_show(save = save, show = show)

    if show is False:
        return clustermap
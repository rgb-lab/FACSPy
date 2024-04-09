from anndata import AnnData
import numpy as np
import pandas as pd

import seaborn as sns
from typing import Literal, Optional, Union

from ._utils import (_scale_data,
                     _map_obs_to_cmap,
                     _calculate_sample_distance,
                     _calculate_linkage,
                     _prepare_heatmap_data,
                     _remove_dendrogram,
                     _add_metaclusters,
                     _remove_ticklabels,
                     _remove_ticks,
                     _scale_cbar_to_heatmap,
                     _add_categorical_legend_to_clustermap,
                     _calculate_correlation_data,
                     add_annotation_plot,
                     ANNOTATION_CMAPS,
                     savefig_or_show)
from ._clustermap import create_clustermap

from .._utils import (_default_gate_and_default_layer,
                      _enable_gate_aliases,
                      _fetch_fluo_channels)

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
@_enable_gate_aliases
def expression_heatmap(adata: AnnData,
                       gate: str,
                       layer: str,
                       metadata_annotation: Optional[Union[list[str], str]] = None,
                       marker_annotation: Optional[str] = None,
                       include_technical_channels: bool = False,
                       exclude: Optional[Union[list[str], str]] = None,
                       data_group: str = "sample_ID",
                       data_metric: Literal["mfi", "fop"] = "mfi",
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cluster_method: Literal["correlation", "distance"] = "distance",
                       cmap: str = "RdBu_r",
                       metaclusters: Optional[int] = None,
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc",
                       y_label_fontsize: Optional[Union[int, float]] = 10,
                       figsize: Optional[tuple[float, float]] = (5,3.8),
                       return_dataframe: bool = False,
                       return_fig: bool = False,
                       show: bool = True,
                       save: Optional[str] = None
                       ) -> Optional[Union[sns.matrix.ClusterGrid, pd.DataFrame]]:
    """\
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
    metadata_annotation
        Controls the annotated variables on top of the plot.
    marker_annotation
        creates a second plot on top of the heatmap where marker expressions can
        be shown. 
    include_technical_channels
        Whether to include technical channels. If set to False, will exclude
        all channels that are not labeled with `type=="fluo"` in adata.var.
    exclude
        Channels to be excluded from plotting.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score).
    corr_method
        Correlation method that is used for hierarchical clustering by sample correlation.
        if `cluster_method == distance`, this parameter is ignored. One of `pearson`, `spearman` 
        or `kendall`.
    cluster_method
        Method for hierarchical clustering of displayed samples. If `correlation`, the correlation
        specified by corr_method is computed (default: pearson). If `distance`, the euclidean
        distance is computed.
    cmap
        Sets the colormap for plotting the markers
    metaclusters
        controls the n of metaclusters to be computed
    label_metaclusters_in_dataset
        Whether to label the calculated metaclusters and write into the metadata
    label_metaclusters_key
        Column name that is used to store the metaclusters in
    y_label_fontsize
        controls the fontsize of the marker labels
    figsize
        Contains the dimensions of the final figure as a tuple of two ints or floats.
    return_dataframe
        If set to True, returns the raw data that are used for plotting as a dataframe.
    return_fig
        If set to True, the figure is returned.
    show
        Whether to show the figure. Defaults to True.
    save
        Expects a file path including the file name.
        Saves the figure to the indicated path. Defaults to None.

    Returns
    -------
    If `show==False` a :class:`~seaborn.ClusterGrid`
    If `return_fig==True` a :class:`~seaborn.ClusterGrid`
    If `return_dataframe==True` a :class:`~pandas.DataFrame` containing the data used for plotting

    Examples
    --------

    .. plot::
        :context: close-figs

        import FACSPy as fp

        dataset = fp.mouse_lineages()
        
        fp.tl.mfi(dataset, layer = "transformed")

        fp.pl.expression_heatmap(
            dataset,
            gate = "CD45+",
            layer = "transformed",
            metadata_annotation = ["organ", "sex"],
            marker_annotation = "B220"
        )

    """

    if not isinstance(metadata_annotation, list) and metadata_annotation is not None:
        metadata_annotation = [metadata_annotation]
    elif metadata_annotation is None:
        metadata_annotation = []

    if not isinstance(exclude, list):
        if exclude is None:
            exclude = []
        else:
            exclude = [exclude]

    raw_data, plot_data = _prepare_heatmap_data(adata = adata,
                                                gate = gate,
                                                layer = layer,
                                                data_metric = data_metric,
                                                data_group = data_group,
                                                include_technical_channels = include_technical_channels,
                                                exclude = exclude,
                                                scaling = scaling,
                                                return_raw_data = True)
    
    plot_data = plot_data.dropna(axis = 0, how = "any")

    cols_to_plot = _fetch_fluo_channels(adata) if not include_technical_channels else adata.var_names.tolist()
    assert isinstance(exclude, list)
    cols_to_plot = [col for col in cols_to_plot if col not in exclude]
    
    if cluster_method == "correlation":
        col_linkage = _calculate_linkage(
            _calculate_correlation_data(
                plot_data[cols_to_plot].T, corr_method
                )
            )
    
    elif cluster_method == "distance":
        col_linkage = _calculate_linkage(
            _calculate_sample_distance(
                plot_data[cols_to_plot]
                )
            )

    if metaclusters is not None:
        metadata_annotation += ["metacluster"]
        plot_data = _add_metaclusters(adata = adata,
                                      data = plot_data,
                                      row_linkage = col_linkage,
                                      n_clusters = metaclusters,
                                      sample_IDs = raw_data["sample_ID"],
                                      label_metaclusters = label_metaclusters_in_dataset,
                                      label_metaclusters_key = label_metaclusters_key)

    if return_dataframe:
        return plot_data

    plot_data = plot_data.set_index(data_group)
    
    if metadata_annotation:
        col_colors = [
            _map_obs_to_cmap(plot_data, group, ANNOTATION_CMAPS[i])
            for i, group in enumerate(metadata_annotation)
        ]
    else:
        col_colors = None

    ### for the heatmap, the dataframe is transposed so that sample_IDs are the columns
    clustermap = create_clustermap(data = plot_data[cols_to_plot].T,
                                   col_colors = col_colors,
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

    if metadata_annotation:
        _add_categorical_legend_to_clustermap(clustermap,
                                              heatmap = ax,
                                              data = plot_data,
                                              annotate = metadata_annotation)

    if marker_annotation is not None:
        if marker_annotation in adata.var_names:
            raw_data = raw_data.set_index(data_group)
            annot_frame = raw_data[marker_annotation]

        add_annotation_plot(adata = adata,
                            annotate = marker_annotation,
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

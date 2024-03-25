import pandas as pd
from anndata import AnnData

import seaborn as sns
from typing import Literal, Union, Optional

from ._utils import (_map_obs_to_cmap,
                     _append_metadata,
                     _prepare_heatmap_data,
                     _calculate_linkage,
                     _add_metaclusters,
                     _remove_ticklabels,
                     _remove_ticks,
                     _scale_cbar_to_heatmap,
                     _add_categorical_legend_to_clustermap,
                     _calculate_correlation_data,
                     savefig_or_show,
                     _has_interval_index,
                     ANNOTATION_CMAPS,
                     CONTINUOUS_CMAPS)

from ._clustermap import create_clustermap

from .._utils import _default_gate_and_default_layer, _enable_gate_aliases

def _calculate_correlations(adata: AnnData,
                            plot_data: pd.DataFrame,
                            corr_method: Literal["pearson", "kendall", "spearman"]) -> pd.DataFrame:
    sample_IDs = plot_data["sample_ID"].tolist()
    channels = [col for col in plot_data.columns if col in adata.var_names]
    correlations = _calculate_correlation_data(plot_data[channels].T,
                                               corr_method = corr_method)
    plot_data = pd.DataFrame(data = correlations.values,
                             columns = sample_IDs,
                             index = sample_IDs)
    plot_data = plot_data.fillna(0)
    plot_data["sample_ID"] = sample_IDs
    plot_data = _append_metadata(adata, plot_data)
    return plot_data

@_default_gate_and_default_layer
@_enable_gate_aliases
def sample_correlation(adata: AnnData,
                       gate: str,
                       layer: str,
                       metadata_annotation: Optional[Union[list[str], str]] = None,
                       include_technical_channels: bool = False,
                       data_group: str = "sample_ID",
                       data_metric: Literal["mfi", "fop"] = "mfi",
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = "MinMaxScaler",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: Optional[str] = "inferno",
                       metaclusters: Optional[int] = None,
                       label_metaclusters_in_dataset: bool = True,
                       label_metaclusters_key: Optional[str] = "metacluster_sc",
                       figsize: tuple[float, float] = (4,4),
                       return_dataframe: bool = False,
                       return_fig: bool = False,
                       show: bool = True,
                       save: Optional[str] = None
                       ) -> Optional[Union[sns.matrix.ClusterGrid, pd.DataFrame]]:
    """\
    Plot for sample correlation.

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
    include_technical_channels
        Whether to include technical channels. If set to False, will exclude
        all channels that are not labeled with `type=="fluo"` in adata.var.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe.
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively.
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score). Defaults to MinMaxScaler.
    corr_method
        Correlation method that is used for correlation analysis. One of
        `pearson`, `spearman` or `kendall`. Defaults to `pearson`.
    cmap
        Sets the colormap for plotting the markers.
    metaclusters
        Controls the n of metaclusters to be computed.
    label_metaclusters_in_dataset
        Whether to label the calculated metaclusters and write into the metadata.
    label_metaclusters_key
        Column name that is used to store the metaclusters in.
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

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.tl.mfi(dataset)
    >>> fp.pl.sample_correlation(
    ...     dataset,
    ...     gate = "live",
    ...     layer = "transformed",
    ...     metadata_annotation = ["condition", "sex"]
    ... )

    """

    plot_data = _prepare_heatmap_data(adata = adata,
                                      gate = gate,
                                      layer = layer,
                                      data_metric = data_metric,
                                      data_group = data_group,
                                      include_technical_channels = include_technical_channels,
                                      scaling = scaling)
    plot_data = _calculate_correlations(adata = adata,
                                        plot_data = plot_data,
                                        corr_method = corr_method)
 
    if not isinstance(metadata_annotation, list) and metadata_annotation is not None:
        metadata_annotation = [metadata_annotation]
    elif metadata_annotation is None:
        metadata_annotation = []

    row_linkage = _calculate_linkage(plot_data[plot_data["sample_ID"].to_list()])

    if metaclusters is not None:
        metadata_annotation += ["metacluster"]
        plot_data = _add_metaclusters(adata = adata,
                                      data = plot_data,
                                      row_linkage = row_linkage,
                                      n_clusters = metaclusters,
                                      sample_IDs = plot_data["sample_ID"],
                                      label_metaclusters = label_metaclusters_in_dataset,
                                      label_metaclusters_key = label_metaclusters_key)

    if return_dataframe:
        return plot_data

    if metadata_annotation:
        row_colors = [
            _map_obs_to_cmap(plot_data,
                    group,
                    CONTINUOUS_CMAPS[i] if _has_interval_index(plot_data[group]) else ANNOTATION_CMAPS[i]
                    )
            for i, group in enumerate(metadata_annotation)
        ]
        col_colors = row_colors
    else:
        row_colors = None
        col_colors = None

    clustermap = create_clustermap(data = plot_data[plot_data["sample_ID"].to_list()],
                                   row_colors = row_colors,
                                   col_colors = col_colors,
                                   row_linkage = row_linkage,
                                   col_linkage = row_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   vmin = -1,
                                   vmax = 1,
                                   cbar_kws = {"label": f"{corr_method} correlation",
                                               "orientation": 'horizontal'}
                                   )
    ax = clustermap.ax_heatmap
    heatmap_position = ax.get_position()
    
    _scale_cbar_to_heatmap(clustermap,
                          heatmap_position = heatmap_position)
    _remove_ticklabels(ax, which = "both")
    _remove_ticks(ax, which = "both")
    _add_categorical_legend_to_clustermap(clustermap,
                                          heatmap = ax,
                                          data = plot_data,
                                          annotate = metadata_annotation)
    if return_fig:
        return clustermap
    savefig_or_show(save = save, show = show)
    if show is False:
        return clustermap

import pandas as pd
from anndata import AnnData

import seaborn as sns
from typing import Literal, Union, Optional
from ._utils import (_map_obs_to_cmap,
                     _calculate_sample_distance,
                     _prepare_heatmap_data,
                     _append_metadata,
                     _calculate_linkage,
                     _add_metaclusters,
                     _remove_ticklabels,
                     _remove_ticks,
                     _scale_cbar_to_heatmap,
                     _add_categorical_legend_to_clustermap,
                     ANNOTATION_CMAPS,
                     CONTINUOUS_CMAPS,
                     _has_interval_index,
                     savefig_or_show)

from ._clustermap import create_clustermap

from .._utils import _default_gate_and_default_layer, _enable_gate_aliases
from ..exceptions._exceptions import InfRemovalWarning

def _calculate_distances(adata: AnnData,
                         plot_data: pd.DataFrame) -> pd.DataFrame:
    sample_IDs = plot_data["sample_ID"].tolist()
    channels = [col for col in plot_data.columns if col in adata.var_names]

    if plot_data[channels].isna().any(axis = None):
        InfRemovalWarning("NA Values were found and will be removed.")
        plot_data = plot_data.dropna(how = "any")

    sample_distances = _calculate_sample_distance(plot_data[channels])
    plot_data = pd.DataFrame(data = sample_distances,
                             columns = sample_IDs,
                             index = sample_IDs)

    plot_data = plot_data.fillna(0)
    plot_data["sample_ID"] = sample_IDs
    plot_data = _append_metadata(adata, plot_data)
    # plot_data = plot_data.dropna(how = "any")
    return plot_data


@_default_gate_and_default_layer
@_enable_gate_aliases
def sample_distance(adata: AnnData,
                    gate: str,
                    layer: str,
                    metadata_annotation: Optional[Union[list[str], str]] = None,
                    include_technical_channels: bool = False,
                    exclude: Optional[Union[list[str], str]] = None,
                    data_group: str = "sample_ID",
                    data_metric: Literal["mfi", "fop"] = "mfi",
                    scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                    cmap: Optional[str] = "inferno",
                    metaclusters: Optional[int] = None,
                    label_metaclusters_in_dataset: bool = True,
                    label_metaclusters_key: Optional[str] = "sample_distance_metaclusters",
                    figsize: tuple[float, float] = (4,4),
                    return_dataframe: bool = False,
                    return_fig: bool = False,
                    show: bool = True,
                    save: Optional[str] = None
                    ) -> Optional[Union[sns.matrix.ClusterGrid, pd.DataFrame]]:
    """\
    Plot to display sample to sample distance as a heatmap.

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
    exclude
        Channels to be excluded from plotting.
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively.
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score)
    cmap
        Sets the colormap for plotting the markers
    metaclusters
        Controls the n of metaclusters to be computed
    label_metaclusters_in_dataset
        Whether to label the calculated metaclusters and write into the metadata
    label_metaclusters_key
        Column name that is used to store the metaclusters in
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
        
        fp.tl.mfi(dataset, layer = "compensated")

        fp.pl.sample_distance(
            dataset,
            gate = "CD45+",
            layer = "compensated",
            metadata_annotation = ["organ", "sex"]
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

    plot_data = _prepare_heatmap_data(adata = adata,
                                      gate = gate,
                                      layer = layer,
                                      data_metric = data_metric,
                                      data_group = data_group,
                                      include_technical_channels = include_technical_channels,
                                      exclude = exclude,
                                      scaling = scaling)

    plot_data = _calculate_distances(adata = adata,
                                     plot_data = plot_data)

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
    
    clustermap = create_clustermap(data = plot_data[plot_data["sample_ID"]],
                                   row_colors = row_colors,
                                   col_colors = col_colors,
                                   row_linkage = row_linkage,
                                   col_linkage = row_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   cbar_kws = {"label": "distance",
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

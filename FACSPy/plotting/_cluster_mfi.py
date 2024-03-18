from anndata import AnnData
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Literal, Optional, Union

from ._utils import (_scale_data,
                     _calculate_sample_distance,
                     _calculate_linkage,
                     _prepare_heatmap_data,
                     _get_uns_dataframe,
                     _scale_cbar_to_heatmap,
                     _calculate_correlation_data,
                     _remove_dendrogram,
                     add_annotation_plot,
                     savefig_or_show)
from ._clustermap import create_clustermap
from ._frequency_plots import _prep_cluster_abundance
from ._categorical_stripplot import _categorical_strip_box_plot

from .._utils import (_default_gate_and_default_layer,
                      _fetch_fluo_channels,
                      _enable_gate_aliases)
from .._settings import settings

def _cluster_mfi_fop_baseplot(data_metric: str,
                              adata: AnnData,
                              gate: str,
                              layer: str,
                              cluster_key: str,
                              marker: str,
                              splitby: Optional[str],
                              cmap: str,
                              order: list[str],
                              stat_test: str,
                              figsize: tuple[float, float],
                              return_dataframe: bool,
                              return_fig: bool,
                              ax: Axes,
                              save: bool,
                              show: bool):

    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{cluster_key}_{layer}")

    if "sample_ID" not in data.columns and "sample_ID" not in data.index.names:
        raise TypeError(f"Please rerun cluster MFI analysis fp.tl.{data_metric} without aggregate set to True")
    
    if return_dataframe:
        return data
    
    plot_params = {
        "data": data,
        "x": cluster_key,
        "y": marker,
        "hue": splitby,
        "palette": cmap or settings.default_categorical_cmap if splitby else None,
        "order": order
    }

    fig, ax = _categorical_strip_box_plot(ax = ax,
                                          data = data,
                                          plot_params = plot_params,
                                          groupby = cluster_key,
                                          splitby = splitby,
                                          stat_test = stat_test,
                                          figsize = figsize)

    ax.set_title(f"{marker}\ngrouped by {cluster_key}")
    ax.set_xlabel("")
    ax.set_ylabel(f"{marker} {data_metric.upper()} " +
                  f"[{'AFU' if data_metric == 'mfi' else 'dec.'}]")

    if return_fig:
        return fig

    savefig_or_show(save = save, show = show)
    
    if show is False:
        return ax

@_enable_gate_aliases
def cluster_fop(adata: AnnData,
                gate: str = None,
                layer: str = None,
                cluster_key: str = None,
                marker: str = None,
                splitby: Optional[str] = None,
                cmap: str = None,
                order: list[str] = None,
                stat_test: str = "Kruskal",
                figsize: tuple[float, float] = (3,3),
                return_dataframe: bool = False,
                return_fig: bool = False,
                ax: Axes = None,
                save: bool = None,
                show: bool = None):
    
    return _cluster_mfi_fop_baseplot(data_metric = "fop",
                                     adata = adata,
                                     gate = gate,
                                     layer = layer,
                                     cluster_key = cluster_key,
                                     marker = marker,
                                     splitby = splitby,
                                     cmap = cmap,
                                     order = order,
                                     stat_test = stat_test,
                                     figsize = figsize,
                                     return_dataframe = return_dataframe,
                                     return_fig = return_fig,
                                     ax = ax,
                                     save = save,
                                     show = show)

@_enable_gate_aliases
def cluster_mfi(adata: AnnData,
                gate: str = None,
                layer: str = None,
                cluster_key: str = None,
                marker: str = None,
                splitby: Optional[str] = None,
                cmap: str = None,
                order: list[str] = None,
                stat_test: str = "Kruskal",
                figsize: tuple[float, float] = (3,3),
                return_dataframe: bool = False,
                return_fig: bool = False,
                ax: Axes = None,
                save: bool = None,
                show: bool = None):

    return _cluster_mfi_fop_baseplot(data_metric = "mfi",
                                     adata = adata,
                                     gate = gate,
                                     layer = layer,
                                     cluster_key = cluster_key,
                                     marker = marker,
                                     splitby = splitby,
                                     cmap = cmap,
                                     order = order,
                                     stat_test = stat_test,
                                     figsize = figsize,
                                     return_dataframe = return_dataframe,
                                     return_fig = return_fig,
                                     ax = ax,
                                     save = save,
                                     show = show)

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
def cluster_heatmap(adata: AnnData,
                    gate: str = None,
                    layer: str = None,
                    cluster_key: Optional[Union[str, list[str]]] = None,
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    include_technical_channels: bool = False,
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
    """\
    Plots a heatmap where every column corresponds to one cluster and the rows display the marker expression.

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
        gate parameter, it has a default stored in `fp.settings` which
        can be overwritten by user input.
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
        correlation method that is used for hierarchical clustering by cluster correlation.
        if `cluster_method == distance`, this parameter is ignored. One of `pearson`, `spearman` 
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
    raw_data, plot_data = _prepare_heatmap_data(adata = adata,
                                                gate = gate,
                                                layer = layer,
                                                data_metric = data_metric,
                                                data_group = cluster_key,
                                                include_technical_channels = include_technical_channels,
                                                scaling = scaling,
                                                return_raw_data = True)
    
    if return_dataframe:
        return plot_data

    cols_to_plot = _fetch_fluo_channels(adata) if not include_technical_channels else adata.var_names.tolist()
    plot_data = plot_data.set_index(cluster_key)

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

    clustermap = create_clustermap(data = plot_data[cols_to_plot].T,
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
            annot_frame = _prep_cluster_abundance(
                adata,
                groupby = annotation_kwargs.get("groupby", "sample_ID"),
                cluster_key = annotation_kwargs.get("cluster_key", cluster_key),
                normalize = annotation_kwargs.get("normalize", True),
            )
        elif annotate in adata.var_names:
            raw_data = raw_data.set_index(cluster_key)
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







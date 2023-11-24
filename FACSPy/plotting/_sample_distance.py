import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Literal, Union, Optional
from ._utils import (_scale_data,
                     _map_obs_to_cmap,
                     _calculate_sample_distance,
                     _append_metadata,
                     _calculate_linkage,
                     _add_metaclusters,
                     _remove_ticklabels,
                     _remove_ticks,
                     _scale_cbar_to_heatmap,
                     _add_categorical_legend_to_clustermap,
                     _get_uns_dataframe,
                     ANNOTATION_CMAPS,
                     CONTINUOUS_CMAPS,
                     _has_interval_index,
                     savefig_or_show)

from ._clustermap import create_clustermap

from .._utils import _default_gate_and_default_layer

def _prepare_plot_data(adata: AnnData,
                       raw_data: pd.DataFrame,
                       scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]],
                       copy: bool = False
                       ) -> pd.DataFrame:
    plot_data = raw_data.copy() if copy else raw_data
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names]
    if scaling is not None:
        plot_data[fluo_columns] = _scale_data(plot_data[fluo_columns], scaling)
    sample_distances = _calculate_sample_distance(plot_data[fluo_columns])
    plot_data = pd.DataFrame(data = sample_distances,
                             columns = raw_data["sample_ID"].to_list(),
                             index = raw_data["sample_ID"].to_list())
    plot_data = plot_data.fillna(0)
    plot_data["sample_ID"] = raw_data["sample_ID"].to_list()
    plot_data = _append_metadata(adata, plot_data)

    return plot_data

@_default_gate_and_default_layer
def sample_distance(adata: AnnData,
                    gate: str = None,
                    layer: str = None,

                    annotate: Union[str, list[str]] = None,

                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    
                    scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                    cmap: str = "inferno",
                    figsize: tuple[float, float] = (4,4),
                    return_dataframe: bool = False,
                    metaclusters: Optional[int] = None,
                    label_metaclusters_in_dataset: bool = True,
                    label_metaclusters_key: Optional[str] = "sample_distance_metaclusters",
                    return_fig: bool = False,
                    save: bool = None,
                    show: bool = None) -> Optional[Figure]:
    
    """ plots sample distance data_metrics as a heatmap """
    
    if not isinstance(annotate, list):
        annotate = [annotate] 
    
    raw_data = _get_uns_dataframe(adata = adata,
                                  gate = gate,
                                  table_identifier = f"{data_metric}_{data_group}_{layer}")
    
    plot_data = _prepare_plot_data(adata = adata,
                                   raw_data = raw_data,
                                   copy = False,
                                   scaling = scaling)
    
    if return_dataframe:
        return plot_data
    
    row_linkage = _calculate_linkage(plot_data[plot_data["sample_ID"].to_list()])

    if metaclusters is not None:
        annotate += ["metacluster"]
        plot_data = _add_metaclusters(adata = adata,
                                      data = plot_data,
                                      row_linkage = row_linkage,
                                      n_clusters = metaclusters,
                                      sample_IDs = plot_data["sample_ID"],
                                      label_metaclusters = label_metaclusters_in_dataset,
                                      label_metaclusters_key = label_metaclusters_key)
    
    clustermap = create_clustermap(data = plot_data[plot_data["sample_ID"]],
                                   row_colors = [
                                       _map_obs_to_cmap(plot_data,
                                                        group,
                                                        CONTINUOUS_CMAPS[i] if _has_interval_index(plot_data[group]) else ANNOTATION_CMAPS[i]
                                                        )
                                       for i, group in enumerate(annotate)
                                   ],
                                   col_colors = [
                                       _map_obs_to_cmap(plot_data,
                                                        group,
                                                        CONTINUOUS_CMAPS[i] if _has_interval_index(plot_data[group]) else ANNOTATION_CMAPS[i]
                                                                         )
                                       for i, group in enumerate(annotate)
                                   ],
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
                                          annotate = annotate)
    
    if return_fig:
        return clustermap
    savefig_or_show(save = save, show = show)
    plt.show()
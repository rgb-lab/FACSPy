from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.figure import Figure

from typing import Literal, Optional, Union

<<<<<<< HEAD
from ..utils import subset_gate, find_gate_path_of_gate
=======
from ..utils import find_gate_path_of_gate
>>>>>>> 425dc6a5d3058f2c6623ed6efa4dccb893827e19
from .utils import (scale_data,
                    select_gate_from_multiindex_dataframe,
                    calculate_sample_distance,
                    calculate_linkage,
                    remove_ticklabels,
                    remove_ticks,
                    scale_cbar_to_heatmap,
                    calculate_correlation_data,
                    remove_dendrogram,
<<<<<<< HEAD
                    add_annotation_plot,
                    get_dataframe,
                    ANNOTATION_CMAPS)
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
=======
                    add_annotation_plot)

>>>>>>> 425dc6a5d3058f2c6623ed6efa4dccb893827e19

from ._clustermap import create_clustermap
from ._frequency_plots import prep_dataframe_cluster_freq
from ..exceptions.exceptions import AnalysisNotPerformedError

def cluster_mfi(adata: AnnData,
                marker: Union[str, list[str]],
                groupby: Union[str, list[str]] = None,
                on: Literal["mfi", "fop", "gate_frequency"] = "mfi_c",
                colorby: Optional[str] = None,
                order: list[str] = None,
                gate: str = None,
                overview: bool = False,
                return_dataframe: bool = False) -> Optional[Figure]:

    try:
        data = adata.uns[on]
        data = select_gate_from_multiindex_dataframe(data.T, find_gate_path_of_gate(adata, gate))

    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e

    data.index = data.index.set_names(["cluster", "gate"])
    raw_data = data.reset_index()

    sns.barplot(data = raw_data,
                x = "cluster",
                y = marker)
    plt.show()

def cluster_heatmap(adata: AnnData,
                    groupby: Optional[Union[str, list[str]]],
                    gate: str,
                    scaling: Optional[Literal["MinMaxScaler", "RobustScaler"]] = "MinMaxScaler",
                    on: Literal["mfi", "fop", "gate_frequency"] = "mfi_c",
                    corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                    cluster_method: Literal["correlation", "distance"] = "distance",
                    annotate: Optional[Union[Literal["frequency"], str]] = None,
                    cmap: str = "inferno",
                    return_fig: bool = False,
                    annotation_kwargs: dict = {},
                    figsize: Optional[tuple[int, int]] = (5,3.8),
                    y_label_fontsize: Optional[Union[int, float]] = 4,
                    return_dataframe: bool = False) -> Optional[Figure]:
    
    raw_data = get_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = on,
                             column_identifier_name = "cluster")
    
    plot_data = raw_data.copy()
    fluo_columns = [col for col in raw_data.columns if col in adata.var_names.to_list()]
    if scaling is not None:
        plot_data[fluo_columns] = scale_data(plot_data[fluo_columns], scaling)
    
    plot_data = plot_data[fluo_columns].T

    if return_dataframe:
        return plot_data

    if cluster_method == "correlation":
        col_linkage = calculate_linkage(calculate_correlation_data(plot_data, corr_method))

    elif cluster_method == "distance":
        col_linkage = calculate_linkage(calculate_sample_distance(plot_data.T))

    clustermap = create_clustermap(data = plot_data,
                                   row_cluster = True,
                                   col_linkage = col_linkage,
                                   cmap = cmap,
                                   figsize = figsize,
                                   cbar_kws = {"label": "scaled expression", "orientation": 'vertical'},
                                   vmin = 0 if scaling is not None else None,
                                   vmax = 1 if scaling is not None else None
                                   )
    
    indices = [t.get_text() for t in np.array(clustermap.ax_heatmap.get_xticklabels())]
    
    ax = clustermap.ax_heatmap
    scale_cbar_to_heatmap(clustermap = clustermap,
                          heatmap_position = ax.get_position(),
                          cbar_padding = 1.05,
                          loc = "right")
    remove_ticklabels(ax = ax,
                      which = "x")
    remove_ticks(ax = ax,
                 which = "x")
    
    ax.yaxis.set_ticks_position("left")
    ax.set_yticklabels(ax.get_yticklabels(),
                       fontsize = y_label_fontsize)
    ax.set_ylabel("")
    remove_dendrogram(clustermap, which = "y")
    clustermap.ax_heatmap.set_xlabel("cluster")

    if annotate is not None:
        if annotate == "frequency":
            annot_frame = prep_dataframe_cluster_freq(
                adata,
                groupby = annotation_kwargs.get("groupby", "sample_ID"),
                cluster_key = annotation_kwargs.get("cluster_key", "leiden"),
                normalize = annotation_kwargs.get("normalize", True),
            )
        elif annotate in adata.var_names:
            #raw_data = raw_data.set_index("cluster")
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
    plt.show()







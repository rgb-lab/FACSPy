import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Literal, Union, Optional

from .utils import (prep_uns_dataframe,
                    turn_off_missing_plots,
                    scale_data,
                    select_gate_from_singleindex_dataframe,
                    map_obs_to_cmap)
from ..utils import find_gate_path_of_gate, reduction_names

from ..exceptions.exceptions import AnalysisNotPerformedError


def marker_correlation(adata: AnnData,
                       groupby: Optional[Union[str, list[str]]],
                       gate: str,
                       scaling: Literal["MinMaxScaler", "RobustScaler"] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno") -> Optional[Figure]:
    
    try:
        data = adata.uns[on]
        data = prep_uns_dataframe(adata, data)
        data = select_gate_from_singleindex_dataframe(data, find_gate_path_of_gate(adata, gate))
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]
        data[fluo_columns] = scale_data(data[fluo_columns], scaling)
        
    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e
    
    if not isinstance(groupby, list):
        groupby = [groupby]

    #fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5,5))
    annotation_cmaps = ["Set1", "Set2", "tab10", "hls", "Paired",]
    clustermap = sns.clustermap(data = data[fluo_columns].corr(method = corr_method),
                                #row_colors = [map_obs_to_cmap(data, group, annotation_cmaps[i]) for i, group in enumerate(groupby)],
                                #col_colors = [map_obs_to_cmap(data, group, annotation_cmaps[i]) for i, group in enumerate(groupby)],
                                vmin = -1,
                                vmax = 1,
                                cmap = cmap,
                                dendrogram_ratio = (.1, .1),
                                annot_kws = {"size": 1},
                                figsize = (5,3.8),
                                cbar_kws = {"label": f"{corr_method} correlation",
                                            "orientation": 'horizontal'},
                                #yticklabels = False,
                                xticklabels = False,
                                yticklabels=1
    )
    clustermap.fig.subplots_adjust(right=0.7)

    clustermap.ax_cbar.set_position([0.1, 0, 0.60, 0.02])
    ax = clustermap.ax_heatmap
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 5)

    plt.show()

def sample_correlation(adata: AnnData,
                       groupby: Optional[Union[str, list[str]]],
                       gate: str,
                       scaling: Literal["MinMaxScaler", "RobustScaler"] = "MinMaxScaler",
                       on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                       corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
                       cmap: str = "inferno") -> Optional[Figure]:
    
    try:
        data = adata.uns[on]
        data = prep_uns_dataframe(adata, data)
        data = select_gate_from_singleindex_dataframe(data, find_gate_path_of_gate(adata, gate))
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]
        data[fluo_columns] = scale_data(data[fluo_columns], scaling)
        
    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e
    
    if not isinstance(groupby, list):
        groupby = [groupby]

    #fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (5,5))
    annotation_cmaps = ["Set1", "Set2", "tab10", "hls", "Paired",]
    clustermap = sns.clustermap(data = data[fluo_columns].T.corr(method = corr_method),
                                row_colors = [map_obs_to_cmap(data, group, annotation_cmaps[i]) for i, group in enumerate(groupby)],
                                col_colors = [map_obs_to_cmap(data, group, annotation_cmaps[i]) for i, group in enumerate(groupby)],
                                vmin = -1,
                                vmax = 1,
                                cmap = cmap,
                                dendrogram_ratio = (.1, .1),
                                annot_kws = {"size": 4},
                                figsize = (5,3.8),
                                cbar_kws = {"label": f"{corr_method} correlation",
                                            "orientation": 'horizontal'},
                                yticklabels = False,
                                xticklabels = False)
    clustermap.fig.subplots_adjust(right=0.7)

    clustermap.ax_cbar.set_position([0.16, 0, 0.53, 0.02])
    ax = clustermap.ax_heatmap
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    next_legend = 0.8
    for i, group in enumerate(groupby):
        group_lut = map_obs_to_cmap(data, group, annotation_cmaps[i], return_mapping = True)
        handles = [Patch(facecolor = group_lut[name]) for name in group_lut]
        legend_space = 0.06 * (len(data[group].unique()) + 1)
        group_legend = plt.legend(handles,
                                  group_lut,
                                  title = group,
                                  bbox_to_anchor = (1.01,
                                                    next_legend),
                                  bbox_transform=clustermap.fig.transFigure
                                  )
        next_legend -= legend_space
        clustermap.fig.add_artist(group_legend)

    plt.show()
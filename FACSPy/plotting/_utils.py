import warnings

import matplotlib
from matplotlib.colors import ListedColormap, Normalize, SymLogNorm, LogNorm
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

from scipy.cluster.hierarchy import cut_tree
from scipy.spatial import distance, distance_matrix
from scipy.cluster import hierarchy
from sklearn.preprocessing import (MinMaxScaler,
                                   RobustScaler,
                                   StandardScaler)

from typing import Literal, Union, Optional

from ..dataset.supplements import Metadata
from ..exceptions._exceptions import (AnalysisNotPerformedError,
                                      InvalidScalingError,
                                      MetaclusterOverwriteWarning,
                                      CofactorNotFoundWarning)
from .._utils import find_gate_path_of_gate, scatter_channels

ANNOTATION_CMAPS = ["Set1", "Set2", "tab10", "hls", "Paired"]
CONTINUOUS_CMAPS = ["YlOrRd", "Reds", "YlOrBr", "PuRd", "Oranges", "Greens"]

def _remove_ticks(ax: Axes,
                  which: Literal["x", "y", "both"]) -> None:
    if which == "x":
        ax.set_xticks([])
    elif which == "y":
        ax.set_yticks([])
    elif which == "both":
        ax.set_xticks([])
        ax.set_yticks([])
    return

def _remove_ticklabels(ax: Axes,
                       which: Literal["x", "y", "both"]) -> None:
    if which == "x":
        ax.set_xticklabels([])
    elif which == "y":
        ax.set_yticklabels([])
    elif which == "both":
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return

def _remove_dendrogram(clmap: sns.matrix.ClusterGrid,
                       which: Literal["x", "y", "both"]) -> None:
    if which == "x":
        clmap.ax_col_dendrogram.set_visible(False)
    if which == "y":
        clmap.ax_row_dendrogram.set_visible(False)
    if which == "both":
        clmap.ax_col_dendrogram.set_visible(False)
        clmap.ax_row_dendrogram.set_visible(False)

def _add_metaclusters(adata: AnnData,
                      data: pd.DataFrame,
                      row_linkage: np.ndarray,
                      n_clusters: int,
                      sample_IDs: Union[pd.Index, pd.Series, list[int], list[str]],
                      label_metaclusters: bool,
                      label_metaclusters_key: str
                      ):
    metaclusters = _calculate_metaclusters(row_linkage, n_clusters = n_clusters, sample_IDs = sample_IDs)
    metacluster_mapping = _map_metaclusters_to_sample_ID(metaclusters, sample_IDs)
    data = _merge_metaclusters_into_dataframe(data, metacluster_mapping)
    
    if label_metaclusters:
        _label_metaclusters_in_dataset(adata = adata,
                                       data = data,
                                       label_metaclusters_key = label_metaclusters_key)
    
    return data

def _calculate_metaclusters(linkage: np.ndarray,
                            n_clusters: int,
                            sample_IDs: list[str]) -> dict[int: set[int]]:
    ### stackoverflow https://stackoverflow.com/questions/65034792/print-all-clusters-and-samples-at-each-step-of-hierarchical-clustering-in-python
    linkage_matrix = linkage
    clusters = cut_tree(linkage_matrix,
                        n_clusters = n_clusters)
    # transpose matrix
    clusters = clusters.T
    for row in clusters[::-1]:
        # create empty dictionary
        groups = {}
        for sid, g in zip(sample_IDs, row):
            if g not in groups:
                # add new key to dict and assign empty set
                groups[g] = set([])
            # add to set of certain group
            groups[g].add(sid)

    return groups

def _map_metaclusters_to_sample_ID(metaclusters: dict,
                                   sample_IDs: list) -> pd.DataFrame:
    sample_IDs = pd.DataFrame(sample_IDs, columns = ["sample_ID"])
    for metacluster in metaclusters:
        sample_IDs.loc[sample_IDs["sample_ID"].isin(metaclusters[metacluster]), "metacluster"] = str(int(metacluster))
    return sample_IDs

def _merge_metaclusters_into_dataframe(data: pd.DataFrame,
                                       metacluster_mapping: pd.DataFrame) -> pd.DataFrame:
    if "metacluster" in data.columns:
        warnings.warn("Overwriting metaclusters in dataset. To avoid that, set a label_metaclusters_key. ",
                      MetaclusterOverwriteWarning)
        data = data.drop(["metacluster"],
                          axis = 1)
    return pd.merge(data, metacluster_mapping, on = "sample_ID")

def _label_metaclusters_in_dataset(adata: AnnData,
                                   data: pd.DataFrame,
                                   label_metaclusters_key: Optional[str] = None) -> None:
    metadata: Metadata = adata.uns["metadata"]
    metadata.dataframe = _merge_metaclusters_into_dataframe(data = metadata.dataframe,
                                                            metacluster_mapping = data[["sample_ID", "metacluster"]])

    if label_metaclusters_key is not None:
        if label_metaclusters_key in metadata.dataframe.columns:
            warnings.warn("Overwriting metaclusters in dataset.",
                          MetaclusterOverwriteWarning)
            metadata.dataframe = metadata.dataframe.drop([label_metaclusters_key],
                                                         axis = 1)
        metadata.dataframe.rename(columns = {"metacluster": label_metaclusters_key},
                                  inplace = True)
    
    return

def _calculate_linkage(dataframe: pd.DataFrame) -> np.ndarray:
    """calculates the linkage"""
    return hierarchy.linkage(distance.pdist(dataframe), method='average')

def _calculate_sample_distance(dataframe: pd.DataFrame) -> np.ndarray:
    """ returns sample distance matrix of given dataframe"""
    return distance_matrix(dataframe.to_numpy(),
                           dataframe.to_numpy())

def _remove_unused_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ handles the case where categorical variables are still there that are not present anymore """
    for col in dataframe.columns:
        if isinstance(dataframe[col].dtype, pd.CategoricalDtype):
            dataframe[col] = dataframe[col].cat.remove_unused_categories()
    return dataframe

def _append_metadata(adata: AnnData,
                     dataframe_to_merge: pd.DataFrame) -> pd.DataFrame:
    metadata: pd.DataFrame = adata.uns["metadata"].to_df().copy()

    if any(col in dataframe_to_merge.columns for col in metadata.columns):
        metadata = metadata.drop([col for col in metadata.columns
                                  if col in dataframe_to_merge.columns
                                  and not col == "sample_ID"],
                                 axis = 1)

    return _remove_unused_categories(
        pd.merge(dataframe_to_merge.reset_index(),
                 metadata,
                 on = "sample_ID",
                 how = "outer"
        )
    )

def _get_uns_dataframe(adata: AnnData,
                       gate: str,
                       table_identifier: str) -> pd.DataFrame:
    
    if table_identifier not in adata.uns:
        raise AnalysisNotPerformedError(table_identifier)
    
    data: pd.DataFrame = adata.uns[table_identifier].copy()
    data = data.loc[data.index.get_level_values("gate") == find_gate_path_of_gate(adata, gate),:]
    data = data.reset_index()
    if "sample_ID" in data.columns:
        data = _append_metadata(adata, data)
    return data

def _select_gate_from_multiindex_dataframe(dataframe: pd.DataFrame,
                                           gate: str) -> pd.DataFrame:
    return dataframe.loc[dataframe.index.get_level_values("gate") == gate,:]

def _scale_data(dataframe: pd.DataFrame,
                scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]) -> np.ndarray:
    if scaling is None:
        return dataframe.values
    elif scaling == "MinMaxScaler":
        return MinMaxScaler().fit_transform(dataframe)
    elif scaling == "RobustScaler":
        return RobustScaler().fit_transform(dataframe)
    elif scaling == "StandardScaler":
        return StandardScaler().fit_transform(dataframe)
    raise InvalidScalingError(scaler = scaling)

def _calculate_correlation_data(data: pd.DataFrame,
                                corr_method: Literal["pearson", "spearman", "kendall"]) -> pd.DataFrame:
    return data.corr(method = corr_method)

def add_annotation_plot(adata: AnnData,
                        annotate: str,
                        annot_frame: pd.DataFrame,
                        indices: list[Union[str, int]],
                        clustermap: sns.matrix.ClusterGrid,
                        y_label_fontsize: Optional[int],
                        y_label: Optional[str]
                        ) -> None:
    #TODO: add sampleID/clusterID on top?
    annot_frame = annot_frame.loc[indices]
    divider = make_axes_locatable(clustermap.ax_heatmap)
    ax: Axes = divider.append_axes("top", size = "20%", pad = 0.05)
    annot_frame.plot(kind = "bar",
                     stacked = annotate == "frequency",
                     legend = True,
                     ax = ax,
                     subplots = False,
                     )
    ax.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left")
    ax.set_ylabel(y_label)

    _remove_ticklabels(ax, which = "x")
    _remove_ticks(ax, which = "x")
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.4)
    ax.set_xlabel("")

    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(),
                            fontsize = y_label_fontsize)

def _has_interval_index(column: pd.Series) -> bool:
    return isinstance(column.dtype, pd.CategoricalDtype) and isinstance(column.cat.categories, pd.IntervalIndex)

def _add_categorical_legend_to_clustermap(clustermap: sns.matrix.ClusterGrid,
                                          heatmap: Axes,
                                          data: pd.DataFrame,
                                          annotate: list[str]) -> None:
    next_legend = 0
    for i, group in enumerate(annotate):
        group_lut = _map_obs_to_cmap(data,
                                     group,
                                     CONTINUOUS_CMAPS[i] if _has_interval_index(data[group])
                                                         else ANNOTATION_CMAPS[i],
                                     return_mapping = True)
        if _has_interval_index(data[group]):
            sorted_index = list(data[group].cat.categories.values)
            if np.nan in group_lut.keys():
                sorted_index = [np.nan] + sorted_index
            group_lut = {key: group_lut[key] for key in sorted_index}
        handles = [Patch(facecolor = group_lut[name]) for name in group_lut]
        legend_space = 0.1 * (len(handles) + 1)
        group_legend = heatmap.legend(handles,
                                      group_lut,
                                      title = group,
                                      loc = "upper left",
                                      bbox_to_anchor = (1.02, 1 - next_legend, 0, 0),
                                      bbox_transform = heatmap.transAxes
                                      )

        next_legend += legend_space
        clustermap.fig.add_artist(group_legend)

def _scale_cbar_to_heatmap(clustermap: sns.matrix.ClusterGrid,
                           heatmap_position: Bbox,
                           cbar_padding: Optional[float] = 0.7,
                           cbar_height: Optional[float] = 0.02,
                           loc: Literal["bottom", "right"] = "bottom") -> None:
    if loc == "bottom":
        clustermap.ax_cbar.set_position([heatmap_position.x0,
                                         heatmap_position.y0 * cbar_padding,
                                         heatmap_position.x1 - heatmap_position.x0,
                                         cbar_height])
    if loc == "right":
        clustermap.ax_cbar.set_position([heatmap_position.x1 * cbar_padding,
                                         heatmap_position.y0,
                                         cbar_height,
                                         heatmap_position.x1 - heatmap_position.x0])
    return

def _map_obs_to_cmap(data: pd.DataFrame,
                     parameter_to_map: str,
                     cmap: str = "Set1",
                     return_mapping: bool = False,
                     as_series: bool = True) -> dict[str, tuple[float, float, float]]:
    if _has_interval_index(data[parameter_to_map]):
        obs = list(data[parameter_to_map].cat.categories.values)
        if np.nan in data[parameter_to_map].unique():
            obs = [np.nan] + obs
    else:    
        obs = data[parameter_to_map].unique()
    cmap = sns.color_palette(cmap, len(obs))
    mapping = {obs_entry: cmap[i] for i, obs_entry in enumerate(obs)}
    if return_mapping:
        return mapping
    if as_series:
        return pd.Series(data[parameter_to_map].astype("object").map(mapping), name = parameter_to_map)
    return data[parameter_to_map].astype("object").map(mapping)

def calculate_nrows(ncols: int, 
                    dataset: pd.DataFrame):
    return int(
            np.ceil(
                len(dataset.columns)/ncols
            )
        )

def calculate_fig_size(ncols: int,
                       nrows: int,
                       groupby_list: list = None) -> tuple[int, int]:
    
    x_dim_scale_factor = (1 + (0.07 * len(groupby_list))) if groupby_list is not None else 1
    x_dimension = 2 * ncols * x_dim_scale_factor
    y_dimension = 1.5 * nrows if groupby_list is None else 1.8 * nrows
    return (x_dimension, y_dimension)

def turn_off_missing_plot(ax: Axes) -> Axes:
    ax.axis("off")
    return ax

def turn_off_missing_plots(ax: Axes) -> Axes:
    for axs in ax:
        if not axs.lines and not axs.collections:
            turn_off_missing_plot(axs)
    return ax

def savefig_or_show(show: Optional[bool] = None,
                    dpi: Optional[int] = 300,
                    ext: str = None,
                    save: Union[bool, str, None] = None):
    """
    simple save or show function
    """
    if show is None:
        show = True
    
    if save:
        assert isinstance(save, str)
        plt.savefig(save,
                    dpi = dpi,
                    bbox_inches = "tight")
    
    if show:
        plt.show()
    
    if save:
        plt.close()

def _get_cofactor_from_var(adata: AnnData,
                           channel: str) -> float:
    return float(adata.var.loc[adata.var["pns"] == channel, "cofactors"].iloc[0])

def _color_var_is_categorical(vals: pd.Series) -> bool:
    return isinstance(vals.dtype, pd.CategoricalDtype)

def _is_scatter(channel: str) -> bool:
    return any(k in channel for k in scatter_channels)

def _transform_data_to_scale(data: np.ndarray,
                             channel: str,
                             adata: AnnData,
                             user_scale) -> np.ndarray:
    scale = _define_axis_scale(channel, user_scale=user_scale)
    if scale == "linear":
        return data
    elif scale == "log":
        transformed = np.log10(data)
        # data can be negative to nan would be produced
        # which would mess up the density function
        transformed[np.where(np.isnan(transformed))] = 0.0
        return transformed
    elif scale == "symlog":
        cofactor = _retrieve_cofactor_or_set_to_default(adata,
                                                        channel)
        return np.arcsinh(data / cofactor)

def _define_axis_scale(channel,
                       user_scale: Literal["biex", "linear", "log"]) -> bool:
    """
    decides if data are plotted on a linear or biex scale
    Log Scaling is not a default but has to be set by the user
    explicitly.
    """
    if user_scale:
        if user_scale == "biex":
            return "symlog"
        return user_scale
    if _is_scatter(channel):
        return "linear"
    return "symlog"

def _continous_color_vector(df: pd.DataFrame,
                            color_col: str,
                            vmin: Optional[float],
                            vmax: Optional[float]):
    color_vector = df[color_col].values.copy()
    if vmin:
        color_vector[np.where(color_vector < vmin)] = vmin
    if vmax:
        color_vector[np.where(color_vector > vmax)] = vmax
    return color_vector

def _retrieve_cofactor_or_set_to_default(adata, channel) -> float:
    try:
        return _get_cofactor_from_var(adata, channel)
    except KeyError:
        # which means cofactors were not calculated
        warnings.warn("Cofactor not found. Setting to 1000 for plotting",
                      CofactorNotFoundWarning)
        return 1000

def _get_cbar_normalizations(color_vector: np.ndarray,
                             color_scale: Literal["biex", "log", "linear"],
                             color_cofactor: Optional[float]):
    if color_scale == "biex":
        norm = SymLogNorm(vmin = np.min(color_vector),
                          vmax = np.max(color_vector),
                          linthresh = color_cofactor)
    elif color_scale == "log":
        norm = LogNorm(vmin = np.min(color_vector),
                       vmax = np.max(color_vector))
    else:
        assert color_scale == "linear"
        norm = Normalize(vmin = np.min(color_vector),
                         vmax = np.max(color_vector))
        
    return norm

def _transform_color_to_scale(color_vector: np.ndarray,
                              color_cofactor: Optional[float],
                              color_scale: Literal["biex", "log", "linear"]) -> np.ndarray:
    norm = _get_cbar_normalizations(color_vector,
                                    color_scale,
                                    color_cofactor)
    return norm(color_vector)

def _generate_continous_color_scale(color_vector: np.ndarray,
                                    cmap: str,
                                    color_cofactor: Optional[float],
                                    ax: Axes,
                                    color_scale: Literal["biex", "log", "linear"]):
    custom_cmap = matplotlib.colormaps[cmap]
    custom_colors = custom_cmap(np.linspace(0,1,256))
    norm = _get_cbar_normalizations(color_vector,
                                    color_scale,
                                    color_cofactor)
    sm = plt.cm.ScalarMappable(cmap = ListedColormap(custom_colors),
                               norm = norm)
    return ax.figure.colorbar(sm,
                              ax = ax)
 

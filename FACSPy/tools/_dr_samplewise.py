from anndata import AnnData

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from umap import UMAP

from typing import Optional, Union, Literal

from ..utils import reduction_names

from ..plotting.utils import select_gate_from_multiindex_dataframe, scale_data

def perform_dr(reduction: Literal["PCA", "MDS", "UMAP", "TSNE"],
               data: np.ndarray,
               n_components: int = 3) -> np.ndarray:
    if reduction == "PCA":
        return PCA(n_components = n_components,
                   random_state = 187).fit_transform(data)
    if reduction == "MDS":
        return MDS(n_components = n_components,
                   random_state = 187).fit_transform(data)
    if reduction == "TSNE":
        return TSNE(n_components = n_components,
                   random_state = 187,
                   learning_rate = "auto",
                   init = "pca").fit_transform(data)
    if reduction == "UMAP":
        return UMAP(n_components = n_components,
                   random_state = 187).fit_transform(data)


def perform_samplewise_dr(data: pd.DataFrame,
                          reduction: Literal["PCA", "MDS", "TSNE", "UMAP"],
                          fluo_channels: Union[pd.Index, list[str]],
                          scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]):
    return_data = data.copy()
    return_data = return_data.T
    data = data.loc[fluo_channels, :]
    data = data.T
    gates = data.index.levels[1]
    for gate in gates:
        gate_specific_data = select_gate_from_multiindex_dataframe(data, gate)
        gate_specific_data = scale_data(gate_specific_data, scaling = scaling)
        coords = perform_dr(reduction, gate_specific_data, n_components = 3)
        coord_columns = reduction_names[reduction]
        return_data.loc[(slice(None), gate), coord_columns] = coords

    return return_data.T

def pca_samplewise(adata: AnnData,
                   on: Literal["mfi", "fop", "gate_frequency"],
                   exclude: Optional[Union[str, list, str]] = [],
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "StandardScaler"):
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    adata.uns[on] = perform_samplewise_dr(adata.uns[on],
                                          reduction = "PCA",
                                          fluo_channels = fluo_channels,
                                          scaling = scaling)


def tsne_samplewise(adata: AnnData,
                   on: Literal["mfi", "fop", "gate_frequency"],
                   exclude: Optional[Union[str, list, str]] = [],
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "StandardScaler"):
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    adata.uns[on] = perform_samplewise_dr(adata.uns[on],
                                          reduction = "TSNE",
                                          fluo_channels = fluo_channels,
                                          scaling = scaling)

def umap_samplewise(adata: AnnData,
                   on: Literal["mfi", "fop", "gate_frequency"],
                   exclude: Optional[Union[str, list, str]] = [],
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "StandardScaler"):
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    
    adata.uns[on] = perform_samplewise_dr(adata.uns[on],
                                          reduction = "UMAP",
                                          fluo_channels = fluo_channels,
                                          scaling = scaling)

def mds_samplewise(adata: AnnData,
                   on: Literal["mfi", "fop", "gate_frequency"],
                   exclude: Optional[Union[str, list, str]] = [],
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "StandardScaler"):
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    
    adata.uns[on] = perform_samplewise_dr(adata.uns[on],
                                          reduction = "MDS",
                                          fluo_channels = fluo_channels,
                                          scaling = scaling)

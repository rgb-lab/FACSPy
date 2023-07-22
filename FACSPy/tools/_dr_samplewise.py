from anndata import AnnData

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from umap import UMAP

from typing import Optional, Union, Literal

from ..utils import reduction_names

from ..plotting.utils import scale_data, select_gate_from_multiindex_dataframe

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
    data = data.loc[:, fluo_channels]
    gates = data.index.get_level_values("gate").unique()
    for gate in gates:
        gate_specific_data = select_gate_from_multiindex_dataframe(data, gate)
        gate_specific_data = scale_data(gate_specific_data, scaling = scaling)
        coords = perform_dr(reduction, gate_specific_data, n_components = 3)
        coord_columns = reduction_names[reduction]
        return_data.loc[return_data.index.get_level_values("gate") == gate, coord_columns] = coords

    return return_data

def pca_samplewise(adata: AnnData,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   data_origin: Literal["compensated", "transformed"] = "compensated",
                   exclude: Optional[Union[str, list, str]] = None,
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler"):
    exclude = [] if exclude is None else exclude
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    table_identifier = f"{data_metric}_{data_group}_{data_origin}"
    adata.uns[table_identifier] = perform_samplewise_dr(adata.uns[table_identifier],
                                                        reduction = "PCA",
                                                        fluo_channels = fluo_channels,
                                                        scaling = scaling)
    
    save_samplewise_dr_settings(adata = adata,
                                data_group = data_group,
                                data_metric = data_metric,
                                data_origin = data_origin,
                                exclude = exclude,
                                scaling = scaling,
                                reduction = "pca")


def tsne_samplewise(adata: AnnData,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    data_origin: Literal["compensated", "transformed"] = "compensated",
                    exclude: Optional[Union[str, list, str]] = None,
                    scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler"):
    exclude = [] if exclude is None else exclude
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    table_identifier = f"{data_metric}_{data_group}_{data_origin}"
    adata.uns[table_identifier] = perform_samplewise_dr(adata.uns[table_identifier],
                                                        reduction = "TSNE",
                                                        fluo_channels = fluo_channels,
                                                        scaling = scaling)
    
    save_samplewise_dr_settings(adata = adata,
                                data_group = data_group,
                                data_metric = data_metric,
                                data_origin = data_origin,
                                exclude = exclude,
                                scaling = scaling,
                                reduction = "tsne")

def umap_samplewise(adata: AnnData,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    data_origin: Literal["compensated", "transformed"] = "compensated",
                    exclude: Optional[Union[str, list, str]] = None,
                    scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler"):
    exclude = [] if exclude is None else exclude
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    table_identifier = f"{data_metric}_{data_group}_{data_origin}"
    
    adata.uns[table_identifier] = perform_samplewise_dr(adata.uns[table_identifier],
                                                        reduction = "UMAP",
                                                        fluo_channels = fluo_channels,
                                                        scaling = scaling)
    
    save_samplewise_dr_settings(adata = adata,
                                data_group = data_group,
                                data_metric = data_metric,
                                data_origin = data_origin,
                                exclude = exclude,
                                scaling = scaling,
                                reduction = "umap")

def mds_samplewise(adata: AnnData,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   data_origin: Literal["compensated", "transformed"] = "compensated",
                   exclude: Optional[Union[str, list, str]] = None,
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler"):
    exclude = [] if exclude is None else exclude
    fluo_channels = [channel for channel in adata.var_names.to_list() if channel not in exclude]
    table_identifier = f"{data_metric}_{data_group}_{data_origin}"
    adata.uns[table_identifier] = perform_samplewise_dr(adata.uns[table_identifier],
                                                        reduction = "MDS",
                                                        fluo_channels = fluo_channels,
                                                        scaling = scaling)
    
    save_samplewise_dr_settings(adata = adata,
                                data_group = data_group,
                                data_metric = data_metric,
                                data_origin = data_origin,
                                exclude = exclude,
                                scaling = scaling,
                                reduction = "mds")

def save_samplewise_dr_settings(adata: AnnData,
                                data_group,
                                data_metric,
                                data_origin,
                                exclude,
                                scaling,
                                reduction):
    if "settings" not in adata.uns:
        adata.uns["settings"] = {}

    adata.uns["settings"][f"_{reduction}_samplewise"] = {
        "data_group": data_group,
        "data_metric": data_metric,
        "data_origin": data_origin,
        "exclude": exclude,
        "scaling": scaling,
    }
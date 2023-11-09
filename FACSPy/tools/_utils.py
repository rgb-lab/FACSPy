from anndata import AnnData
import numpy as np
import pandas as pd
from typing import Literal, Optional

from scipy.sparse import csr_matrix

from .._utils import (contains_only_fluo,
                     subset_fluo_channels,
                     remove_channel,
                     subset_gate)

def assemble_dataframe(adata: AnnData,
                       on: Literal["transformed", "compensated"] = "compensated",
                       expression_data: bool = True) -> pd.DataFrame:
    obs = adata.obs.copy()
    gates = pd.DataFrame(data = adata.obsm["gating"].todense(),
                         columns = adata.uns["gating_cols"].to_list(),
                         index = obs.index)
    if expression_data:
        expression_data = adata.to_df(layer = on)
        return pd.concat([gates, expression_data, obs], axis = 1)
    return pd.concat([gates, obs], axis = 1)

def scale_adata(adata: AnnData,
                scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]]) -> AnnData:
    if scaling == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        adata.X = MinMaxScaler().fit_transform(adata.X)
    elif scaling == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        adata.X = RobustScaler().fit_transform(adata.X)
    else:
        from sklearn.preprocessing import StandardScaler
        adata.X = StandardScaler().fit_transform(adata.X)
    return adata

def preprocess_adata(adata: AnnData,
                     gate: str,
                     data_origin: Literal["compensated", "transformed"],
                     use_only_fluo: bool = True,
                     exclude: Optional[list[str]] = None,
                     scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None) -> AnnData:
    adata = adata.copy()
    adata.X = adata.layers[data_origin]
    
    if scaling is not None:
        adata = scale_adata(adata,
                            scaling = scaling)
        
    if not contains_only_fluo(adata) and use_only_fluo:
        subset_fluo_channels(adata = adata)
    
    if exclude is not None:
        for channel in exclude:
            remove_channel(adata,
                           channel = channel,
                           copy = False)
    
    adata = subset_gate(adata = adata,
                        gate = gate,
                        as_view = True)
    assert adata.is_view
    return adata

def add_array_into_dataframe(dataframe: pd.DataFrame,
                             array: np.ndarray,
                             colnames: list[str]) -> pd.DataFrame:
    dataframe[colnames] = array
    return dataframe

def add_array_into_var(var_frame: pd.DataFrame,
                       array: np.ndarray,
                       colnames: list[str]) -> pd.DataFrame:
    return add_array_into_dataframe(var_frame,
                                    array,
                                    colnames)
def add_array_into_obs(obs_frame: pd.DataFrame,
                       array: np.ndarray,
                       colnames: list[str]) -> pd.DataFrame:
    return add_array_into_dataframe(obs_frame,
                                    array,
                                    colnames)

def merge_adata_frames(adata_frame: pd.DataFrame,
                       subset_frame: pd.DataFrame) -> pd.DataFrame:
    merged = adata_frame.merge(subset_frame,
                               left_index = True,
                               right_index = True,
                               how = "outer")
    return merged.loc[adata_frame.index,:]

def merge_var_frames(adata_var_frame: pd.DataFrame,
                     gate_subset_var_frame: pd.DataFrame) -> pd.DataFrame:
    return merge_adata_frames(adata_var_frame,
                              gate_subset_var_frame)

def merge_obs_frames(adata_obs_frame: pd.DataFrame,
                     gate_subset_obs_frame: pd.DataFrame) -> pd.DataFrame:
    return merge_adata_frames(adata_obs_frame,
                              gate_subset_obs_frame)

def add_uns_data(adata: AnnData,
                 gate_subset: AnnData,
                 old_key: str,
                 key_added: str) -> AnnData:
    adata.uns[key_added] = gate_subset.uns[old_key]
    return adata

def add_array_to_adata_slot(adata: AnnData,
                            data: pd.DataFrame,
                            colnames: list[str],
                            slot: Literal["varm", "obsm", "uns"],
                            key_added: str) -> AnnData:
    if slot == "obsm":
        adata.obsm[key_added] = data[colnames].to_numpy()
    else:
        adata.varm[key_added] = data[colnames].to_numpy()
    
    return adata

def merge_symmetrical_csr_matrix(adata: AnnData,
                                 gate_subset: AnnData,
                                 key_added: str,
                                 old_key) -> csr_matrix:
    
    matrix_frame = pd.DataFrame(data = gate_subset.obsp[f"{key_added}_{old_key}"].toarray(),
                                columns = gate_subset.obs_names,
                                index = gate_subset.obs_names)
    
    semi_full_frame = merge_adata_frames(adata.obs,
                                         matrix_frame)
    
    full_frame = merge_adata_frames(adata.obs,
                                    semi_full_frame[gate_subset.obs_names].T)
    
    full_frame = full_frame[adata.obs_names]
    full_frame = full_frame.fillna(0, inplace = False)
    
    assert full_frame.shape == (adata.shape[0], adata.shape[0])
    
    return csr_matrix(full_frame.loc[adata.obs_names,adata.obs_names])


def merge_neighbors_info_into_adata(adata: AnnData,
                                    gate_subset: AnnData,
                                    neighbors_key: str) -> AnnData:
    adata.uns[neighbors_key] = gate_subset.uns[neighbors_key]
    adata.obsp[f"{neighbors_key}_connectivities"] = merge_symmetrical_csr_matrix(adata = adata,
                                                                                 gate_subset = gate_subset,
                                                                                 key_added = neighbors_key,
                                                                                 old_key = "connectivities")
    adata.obsp[f"{neighbors_key}_distances"] = merge_symmetrical_csr_matrix(adata = adata,
                                                                            gate_subset = gate_subset,
                                                                            key_added = neighbors_key,
                                                                            old_key = "distances")

    return adata

def merge_cluster_info_into_adata(adata: AnnData,
                                  gate_subset: AnnData,
                                  cluster_key: str,
                                  cluster_assignments) -> AnnData:
    if cluster_key in adata.obs.columns:
        adata.obs = adata.obs.drop(cluster_key, axis = 1)
    preprocessed_obs = gate_subset.obs.copy()
    preprocessed_obs[cluster_key] = cluster_assignments
    adata.obs = merge_adata_frames(adata.obs,
                                   preprocessed_obs[[cluster_key]])
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    return adata


def merge_pca_info_into_adata(adata: AnnData,
                              gate_subset: AnnData,
                              dimred_key: str,
                              uns_key: str) -> AnnData:
        
    adata = merge_dimred_varm_info_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                dimred = "pca",
                                                dimred_key = dimred_key)
    
    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                dimred = "pca",
                                                dimred_key = dimred_key)
    
    adata = add_uns_data(adata,
                            gate_subset,
                            old_key = "pca",
                            key_added = f"{uns_key}_pca")

    return adata

def merge_dimred_varm_info_into_adata(adata: AnnData,
                                      gate_subset: AnnData,
                                      dimred: Literal["umap", "tsne", "pca", "diffmap"],
                                      dimred_key: str) -> AnnData:
    if dimred == "pca":
        varm_info = gate_subset.varm["PCs"]
    varm_info_columns = [f"{dimred}{i}" for i in range(1, varm_info.shape[1] + 1)]
    gate_subset_var_frame = gate_subset.var.copy()
    gate_subset_var_frame = add_array_into_var(gate_subset_var_frame,
                                               varm_info,
                                               colnames = varm_info_columns)
    adata_var_frame = adata.var.copy()
    combined_var_frame = merge_var_frames(adata_var_frame = adata_var_frame,
                                          gate_subset_var_frame = gate_subset_var_frame)
    adata = add_array_to_adata_slot(adata = adata,
                                    data = combined_var_frame,
                                    colnames = varm_info_columns,
                                    slot = "varm",
                                    key_added = dimred_key)
    return adata

def merge_dimred_coordinates_into_adata(adata: AnnData,
                                        gate_subset: AnnData,
                                        dimred: Literal["umap", "tsne", "pca", "diffmap"],
                                        dimred_key: str) -> AnnData:
    coordinates = gate_subset.obsm[f"X_{dimred}"]
    coordinates_columns = [f"{dimred}{i}" for i in range(1, coordinates.shape[1] + 1)]

    gate_subset_obs_frame = gate_subset.obs.copy()
    gate_subset_obs_frame = add_array_into_obs(gate_subset_obs_frame,
                                               coordinates,
                                               colnames = coordinates_columns)
    adata_obs_frame = adata.obs.copy()
    combined_obs_frame = merge_obs_frames(adata_obs_frame = adata_obs_frame,
                                          gate_subset_obs_frame = gate_subset_obs_frame)
    adata = add_array_to_adata_slot(adata,
                                    combined_obs_frame,
                                    colnames = coordinates_columns,
                                    slot = "obsm",
                                    key_added = f"X_{dimred_key}")
    
    return adata


import scanpy as sc
import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Literal, Optional

from ..utils import (subset_gate,
                     remove_channel,
                     contains_only_fluo,
                     subset_fluo_channels)

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
    return adata_frame.reset_index().merge(subset_frame,
                                           how = "outer").set_index("index")

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


def merge_dimred_info_into_adata(adata: AnnData,
                                 gate_subset: AnnData,
                                 dim_red: Literal["umap", "tsne", "diffmap"],
                                 gate: str) -> AnnData:
    if dim_red == "umap":
        return merge_umap_info_into_adata(adata = adata,
                                          gate_subset = gate_subset,
                                          gate = gate)
    elif dim_red == "tsne":
        return merge_tsne_info_into_adata(adata = adata,
                                          gate_subset = gate_subset,
                                          gate = gate)
    else:
        return merge_diffmap_info_into_adata(adata = adata,
                                             gate_subset = gate_subset,
                                             gate = gate)

def merge_diffmap_info_into_adata(adata: AnnData,
                                  gate_subset: AnnData,
                                  gate: str) -> AnnData:
    adata = merge_pca_info_into_adata(adata = adata,
                                      gate_subset = gate_subset,
                                      gate = gate)
    adata = merge_neighbors_info_into_adata(adata = adata,
                                            gate_subset = gate_subset,
                                            gate = gate)
    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                gate = gate,
                                                dimred = "diffmap")
    adata = add_uns_data(adata = adata,
                         gate_subset = gate_subset,
                         old_key = "diffmap",
                         key_added = f"{gate}_diffmap")
    
    return adata


def merge_tsne_info_into_adata(adata: AnnData,
                               gate_subset: AnnData,
                               gate: str) -> AnnData:
    
    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                gate = gate,
                                                dimred = "tsne")
    
    adata = add_uns_data(adata = adata,
                         gate_subset = gate_subset,
                         old_key = "tsne",
                         key_added = f"{gate}_tsne")
    
    return adata

def merge_umap_info_into_adata(adata: AnnData,
                               gate_subset: AnnData,
                               gate: str) -> AnnData:
    adata = merge_pca_info_into_adata(adata = adata,
                                      gate_subset = gate_subset,
                                      gate = gate)
    adata = merge_neighbors_info_into_adata(adata = adata,
                                            gate_subset = gate_subset,
                                            gate = gate)
    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                gate = gate,
                                                dimred = "umap")
    adata = add_uns_data(adata = adata,
                         gate_subset = gate_subset,
                         old_key = "umap",
                         key_added = f"{gate}_umap")
    
    return adata


def merge_neighbors_info_into_adata(adata: AnnData,
                                    gate_subset: AnnData,
                                    gate: str) -> AnnData:
    key_added = f"{gate}_neighbors"
    adata.uns[key_added] = gate_subset.uns[key_added]
    adata.obsp[f"{key_added}_connectivities"] = gate_subset.uns[f"{key_added}_connectivities"]
    adata.obsp[f"{key_added}_distances"] = gate_subset.uns[f"{key_added}_distances"]

    return adata

def merge_pca_info_into_adata(adata: AnnData,
                              gate_subset: AnnData,
                              gate: str) -> AnnData:
        
        adata = merge_dimred_varm_info_into_adata(adata = adata,
                                                  gate_subset = gate_subset,
                                                  gate = gate,
                                                  dimred = "pca")
        
        adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                    gate_subset = gate_subset,
                                                    gate = gate,
                                                    dimred = "pca")
        
        adata = add_uns_data(adata,
                             gate_subset,
                             old_key = "pca",
                             key_added = f"{gate}_pca")

        return adata

def merge_dimred_varm_info_into_adata(adata: AnnData,
                                      gate_subset: AnnData,
                                      gate: str,
                                      dimred: Literal["umap", "tsne", "pca", "diffmap"]) -> AnnData:
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
                                    key_added = f"{gate}_PCs")
    return adata

def merge_dimred_coordinates_into_adata(adata: AnnData,
                                        gate_subset: AnnData,
                                        gate: str,
                                        dimred: Literal["umap", "tsne", "pca", "diffmap"]) -> AnnData:
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
                                    key_added = f"X_{gate}_{dimred}")
    
    return adata

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
                     exclude: Optional[list[str]] = None,
                     scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None) -> AnnData:
    
    adata.X = adata.layers[data_origin]
    
    if scaling is not None:
        adata = scale_adata(adata,
                            scaling = scaling)
        
    if not contains_only_fluo(adata):
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
        
def diffmap(adata: AnnData,
            gate: str,
            data_origin: Literal["compensated", "transformed"] = "transformed",
            exclude: Optional[list[str]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            diffmap_kwargs: Optional[dict] = {},
            copy: bool = False) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    
    preprocessed_adata = preprocess_adata(adata = adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          exclude = exclude,
                                          scaling = scaling)
    sc.pp.pca(preprocessed_adata,
              random_state = 187)
    sc.pp.neighbors(preprocessed_adata,
                    random_state = 187,
                    key_added = f"{gate}_neighbors")
    sc.tl.diffmap(preprocessed_adata,
                  neighbors_key = f"{gate}_neighbors",
                  **diffmap_kwargs)

    adata = merge_dimred_info_into_adata(adata,
                                         preprocessed_adata,
                                         dim_red = "diffmap",
                                         gate = gate)

    return adata if copy else None

def umap(adata: AnnData,
         gate: str,
         data_origin: Literal["compensated", "transformed"] = "transformed",
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         umap_kwargs: Optional[dict] = {},
         copy: bool = False) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    
    preprocessed_adata = preprocess_adata(adata = adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          exclude = exclude,
                                          scaling = scaling)
   
    sc.pp.pca(preprocessed_adata,
              random_state = 187)
    sc.pp.neighbors(preprocessed_adata,
                    random_state = 187,
                    key_added = f"{gate}_neighbors")
    sc.tl.umap(preprocessed_adata,
               neighbors_key = f"{gate}_neighbors",
               **umap_kwargs)
    
    adata = merge_dimred_info_into_adata(adata,
                                         preprocessed_adata,
                                         dim_red = "umap",
                                         gate = gate)

    return adata if copy else None

def tsne(adata: AnnData,
         gate: str,
         data_origin: Literal["compensated", "transformed"] = "transformed",
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         tsne_kwargs: Optional[dict] = {},
         copy: bool = False) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    
    preprocessed_adata = preprocess_adata(adata = adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          exclude = exclude,
                                          scaling = scaling)
   
    sc.tl.tsne(preprocessed_adata,
               **tsne_kwargs)
    
    adata = merge_dimred_info_into_adata(adata,
                                         preprocessed_adata,
                                         dim_red = "tsne",
                                         gate = gate)

    return adata if copy else None
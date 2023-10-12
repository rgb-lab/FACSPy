import scanpy as sc
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from anndata import AnnData
from typing import Literal, Optional

from .utils import preprocess_adata
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


def merge_dimred_info_into_adata(adata: AnnData,
                                 gate_subset: AnnData,
                                 dim_red: Literal["umap", "tsne", "diffmap"],
                                 dimred_key: str,
                                 uns_key: str,
                                 neighbors_key: Optional[str] = None) -> AnnData:
    if dim_red == "umap":
        return merge_umap_info_into_adata(adata = adata,
                                          gate_subset = gate_subset,
                                          dimred_key = dimred_key,
                                          neighbors_key = neighbors_key,
                                          uns_key = uns_key)
    elif dim_red == "tsne":
        return merge_tsne_info_into_adata(adata = adata,
                                          gate_subset = gate_subset,
                                          dimred_key = dimred_key,
                                          uns_key = uns_key)
    else:
        return merge_diffmap_info_into_adata(adata = adata,
                                             gate_subset = gate_subset,
                                             dimred_key = dimred_key,
                                             neighbors_key = neighbors_key,
                                             uns_key = uns_key)

def merge_diffmap_info_into_adata(adata: AnnData,
                                  gate_subset: AnnData,
                                  dimred_key: str,
                                  neighbors_key: Optional[str],
                                  uns_key: str) -> AnnData:
    adata = merge_pca_info_into_adata(adata = adata,
                                      gate_subset = gate_subset,
                                      dimred_key = f"{uns_key}_pca",
                                      uns_key = uns_key)

    adata = merge_neighbors_info_into_adata(adata = adata,
                                            gate_subset = gate_subset,
                                            neighbors_key = neighbors_key)

    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                dimred = "diffmap",
                                                dimred_key = dimred_key)
    adata = add_uns_data(adata = adata,
                         gate_subset = gate_subset,
                         old_key = "diffmap_evals",
                         key_added = f"{uns_key}_diffmap_evals")
    
    return adata


def merge_tsne_info_into_adata(adata: AnnData,
                               gate_subset: AnnData,
                               dimred_key: str,
                               uns_key: str) -> AnnData:
    
    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                dimred = "tsne",
                                                dimred_key = dimred_key)
    
    adata = add_uns_data(adata = adata,
                         gate_subset = gate_subset,
                         old_key = "tsne",
                         key_added = f"{uns_key}_tsne")
    
    return adata

def merge_umap_info_into_adata(adata: AnnData,
                               gate_subset: AnnData,
                               dimred_key: str,
                               uns_key: str,
                               neighbors_key: str) -> AnnData:
    adata = merge_pca_info_into_adata(adata = adata,
                                      gate_subset = gate_subset,
                                      dimred_key = f"{uns_key}_pca",
                                      uns_key = uns_key)
    
    adata = merge_neighbors_info_into_adata(adata = adata,
                                            gate_subset = gate_subset,
                                            neighbors_key = neighbors_key)
    
    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = gate_subset,
                                                dimred = "umap",
                                                dimred_key = dimred_key)
    
    adata = add_uns_data(adata = adata,
                         gate_subset = gate_subset,
                         old_key = "umap",
                         key_added = f"{uns_key}_umap")
    
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
    
    return csr_matrix(full_frame)


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

def pca(adata: AnnData,
        gate: str,
        data_origin: Literal["compensated", "transformed"] = "transformed",
        use_only_fluo: bool = True,
        exclude: Optional[list[str]] = None,
        scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
        copy: bool = False,
        *args,
        **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata
    
    preprocessed_adata = preprocess_adata(adata = adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)
    
    uns_key = f"{gate}_{data_origin}"
    dimred_key = f"{uns_key}_pca"
    
    sc.pp.pca(preprocessed_adata,
              random_state = 187,
              *args,
              **kwargs)
    
    adata = merge_pca_info_into_adata(adata,
                                      preprocessed_adata,
                                      dimred_key = dimred_key,
                                      uns_key = uns_key)

    return adata if copy else None

def diffmap(adata: AnnData,
            gate: str,
            data_origin: Literal["compensated", "transformed"] = "transformed",
            use_only_fluo: bool = True,
            exclude: Optional[list[str]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            copy: bool = False,
            *args,
            **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata
    
    preprocessed_adata = preprocess_adata(adata = adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)
    
    uns_key = f"{gate}_{data_origin}"
    dimred_key = f"{uns_key}_diffmap"
    neighbors_key = f"{uns_key}_neighbors"
    
    sc.pp.pca(preprocessed_adata,
              random_state = 187)
    sc.pp.neighbors(preprocessed_adata,
                    random_state = 187,
                    key_added = neighbors_key)
    sc.tl.diffmap(preprocessed_adata,
                  neighbors_key = neighbors_key,
                  n_comps = 3,
                  *args,
                  **kwargs)

    adata = merge_dimred_info_into_adata(adata,
                                         preprocessed_adata,
                                         dim_red = "diffmap",
                                         dimred_key = dimred_key,
                                         neighbors_key = neighbors_key,
                                         uns_key = uns_key)

    return adata if copy else None

def umap(adata: AnnData,
         gate: str,
         data_origin: Literal["compensated", "transformed"] = "transformed",
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata
    
    preprocessed_adata = preprocess_adata(adata = adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)
    uns_key = f"{gate}_{data_origin}"
    dimred_key = f"{uns_key}_umap"
    neighbors_key = f"{uns_key}_neighbors"

    sc.pp.pca(preprocessed_adata,
              random_state = 187)
    sc.pp.neighbors(preprocessed_adata,
                    random_state = 187,
                    key_added = neighbors_key)
    sc.tl.umap(preprocessed_adata,
               neighbors_key = neighbors_key,
               n_components = 3,
               *args,
               **kwargs)
    
    adata = merge_dimred_info_into_adata(adata,
                                         preprocessed_adata,
                                         dim_red = "umap",
                                         dimred_key = dimred_key,
                                         neighbors_key = neighbors_key,
                                         uns_key = uns_key)

    return adata if copy else None

def tsne(adata: AnnData,
         gate: str,
         data_origin: Literal["compensated", "transformed"] = "transformed",
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    preprocessed_adata = preprocess_adata(adata = adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)
    
    uns_key = f"{gate}_{data_origin}"
    dimred_key = f"{uns_key}_tsne"

    sc.tl.tsne(preprocessed_adata,
               *args,
               **kwargs)

    adata = merge_dimred_info_into_adata(adata,
                                         preprocessed_adata,
                                         dim_red = "tsne",
                                         dimred_key = dimred_key,
                                         uns_key = uns_key)

    return adata if copy else None
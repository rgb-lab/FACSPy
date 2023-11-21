from anndata import AnnData
import numpy as np
import pandas as pd
from typing import Literal, Optional
import warnings

from scipy.sparse import csr_matrix

from .._utils import (contains_only_fluo,
                      subset_fluo_channels,
                      remove_channel,
                      subset_gate)
from ..exceptions._exceptions import (InsufficientSampleNumberWarning,
                                      DimredSettingModificationWarning)

def _concat_gate_info_and_obs_and_fluo_data(adata: AnnData,
                                            layer: Literal["transformed", "compensated"] = "compensated") -> pd.DataFrame:
    gate_and_obs = _concat_gate_info_and_obs(adata)
    fluo_data = adata.to_df(layer = layer)
    return pd.concat([gate_and_obs, fluo_data], axis = 1)

def _concat_gate_info_and_obs(adata: AnnData) -> pd.DataFrame:
    obs = adata.obs.copy()
    gates = pd.DataFrame(data = adata.obsm["gating"].todense(),
                         columns = adata.uns["gating_cols"].tolist(),
                         index = obs.index)
    return pd.concat([gates, obs], axis = 1)

def _scale_adata(adata: AnnData,
                 layer: Literal["compensated", "transformed"] = "transformed",
                 scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = "MinMaxScaler") -> AnnData:
    if scaling == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        adata.X = MinMaxScaler().fit_transform(adata.layers[layer])
    elif scaling == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        adata.X = RobustScaler().fit_transform(adata.layers[layer])
    else:
        from sklearn.preprocessing import StandardScaler
        adata.X = StandardScaler().fit_transform(adata.layers[layer])
    return adata

def _preprocess_adata(adata: AnnData,
                      gate: str,
                      layer: Literal["compensated", "transformed"],
                      use_only_fluo: bool = True,
                      exclude: Optional[list[str]] = None,
                      scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None) -> AnnData:
    #adata = adata.copy()
    
    # hacky way to ensure that everything else will be an anndata view
    # modifying X in scaling when X is None is not possible.
    # To save memory, we set it to an empty lil_matrix
    # lil_matrix is used to avoid a sparse efficiency warning
    adata.X = np.zeros(adata.shape, dtype = adata.layers[layer].dtype)
    
    adata = subset_gate(adata = adata,
                        gate = gate,
                        as_view = True)
        
    if use_only_fluo and not contains_only_fluo(adata):
        adata = subset_fluo_channels(adata = adata,
                                     as_view = True)
    
    if exclude is not None:
        adata = remove_channel(adata,
                               channel = exclude,
                               as_view = True,
                               copy = False)

    if scaling is not None:
        adata = _scale_adata(adata,
                             layer = layer,
                             scaling = scaling)
    else:
        # if not scaling the .X slot has to be filled regardless
        adata.X = adata.layers[layer]
    
    assert adata.is_view
    assert adata.X.dtype == adata.layers[layer].dtype, adata.X.dtype
    return adata

def _add_array_into_dataframe(dataframe: pd.DataFrame,
                              array: np.ndarray,
                              colnames: list[str]) -> pd.DataFrame:
    dataframe[colnames] = array
    return dataframe

def _add_array_into_var(var_frame: pd.DataFrame,
                        array: np.ndarray,
                        colnames: list[str]) -> pd.DataFrame:
    return _add_array_into_dataframe(var_frame,
                                     array,
                                     colnames)
def _add_array_into_obs(obs_frame: pd.DataFrame,
                        array: np.ndarray,
                        colnames: list[str]) -> pd.DataFrame:
    return _add_array_into_dataframe(obs_frame,
                                     array,
                                     colnames)

def _merge_adata_frames(adata_frame: pd.DataFrame,
                        subset_frame: pd.DataFrame) -> pd.DataFrame:
    merged = adata_frame.merge(subset_frame,
                               left_index = True,
                               right_index = True,
                               how = "outer")
    return merged.loc[adata_frame.index,:]

def _merge_var_frames(adata_var_frame: pd.DataFrame,
                      gate_subset_var_frame: pd.DataFrame) -> pd.DataFrame:
    return _merge_adata_frames(adata_var_frame,
                               gate_subset_var_frame)

def _merge_obs_frames(adata_obs_frame: pd.DataFrame,
                      gate_subset_obs_frame: pd.DataFrame) -> pd.DataFrame:
    return _merge_adata_frames(adata_obs_frame,
                               gate_subset_obs_frame)

def _add_uns_data(adata: AnnData,
                  data: dict,
                  key_added: str) -> AnnData:
    adata.uns[key_added] = data
    return adata

def _add_array_to_adata_slot(adata: AnnData,
                             data: pd.DataFrame,
                             colnames: list[str],
                             slot: Literal["varm", "obsm", "uns"],
                             key_added: str) -> AnnData:
    """adds array to respective anndata slot"""

    # Do not change the np.asanyarray functionality!
    # When calculating TSNE based on arrays that came from
    # AnnData Views, the order seems to be messed up, and
    # the coordinates are not the same as compared to TSNE
    # that are calculated on AnnData arrays. Still trying
    # to figure out a reason and/or file a bug report, 
    # but to this point hardcoding the order to "C" fixes 
    # the issue

    if slot == "obsm":
        _data = data[colnames].values
        adata.obsm[key_added] = np.asanyarray(_data, order = "C")
    else:
        _data = data[colnames].values
        adata.varm[key_added] = np.asanyarray(_data, order = "C")
    
    return adata

def _calculate_pointers(original_cells: list[str],
                        subset_cells: list[str],
                        indptrs: np.ndarray) -> np.ndarray:
    new_pointers = [0]
    current_pointer: int = indptrs[0]
    cells_transferred = 0
    for cell in original_cells:
        if cell not in subset_cells:
            new_pointers.append(current_pointer)
        else:
            pointer = indptrs[cells_transferred+1]
            new_pointers.append(pointer)
            current_pointer = pointer
            cells_transferred += 1
    
    return np.array(new_pointers)

def _merge_symmetrical_csr_matrix(adata: AnnData,
                                  gate_subset: AnnData,
                                  matrix: csr_matrix) -> csr_matrix:

    original_obs_names = adata.obs_names
    subset_obs_names = gate_subset.obs_names
    subset_idxs = np.array([adata.obs.index.get_loc(idx)
                            for idx in subset_obs_names])
    
    matrix_indices = matrix.indices
    matrix_data = matrix.data
    matrix_indptrs = matrix.indptr

    new_indices = np.array([subset_idxs[position]
                            for position in matrix_indices])
    new_pointers = _calculate_pointers(original_obs_names,
                                       subset_obs_names,
                                       matrix_indptrs)
    
    merged_matrix = csr_matrix(
        (matrix_data.copy().ravel(),
         new_indices.copy().ravel(),
         new_pointers.copy().ravel()),
         shape = (adata.shape[0], adata.shape[0])
    )

    merged_matrix.eliminate_zeros()

    assert merged_matrix.shape == (adata.shape[0], adata.shape[0])

    return merged_matrix

def _merge_neighbors_info_into_adata(adata: AnnData,
                                     gate_subset: AnnData,
                                     connectivities: csr_matrix,
                                     distances: csr_matrix,
                                     neighbors_dict: dict,
                                     neighbors_key: str) -> AnnData:
    adata.uns[neighbors_key] = neighbors_dict
    adata.obsp[f"{neighbors_key}_connectivities"] = _merge_symmetrical_csr_matrix(adata = adata,
                                                                                  gate_subset = gate_subset,
                                                                                  matrix = connectivities)
    adata.obsp[f"{neighbors_key}_distances"] = _merge_symmetrical_csr_matrix(adata = adata,
                                                                             gate_subset = gate_subset,
                                                                             matrix = distances)
                                                                             
    return adata

def _merge_cluster_info_into_adata(adata: AnnData,
                                   gate_subset: AnnData,
                                   cluster_key: str,
                                   cluster_assignments) -> AnnData:
    if cluster_key in adata.obs.columns:
        adata.obs = adata.obs.drop(cluster_key, axis = 1)
    preprocessed_obs = gate_subset.obs.copy()
    preprocessed_obs[cluster_key] = cluster_assignments
    adata.obs = _merge_adata_frames(adata.obs,
                                    preprocessed_obs[[cluster_key]])
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    return adata

def _merge_pca_info_into_adata(adata: AnnData,
                               gate_subset: AnnData,
                               coordinates: np.ndarray,
                               components: np.ndarray,
                               settings: dict,
                               variances: dict,
                               dimred_key: str) -> AnnData:
        
    adata = _merge_dimred_varm_info_into_adata(adata = adata,
                                               gate_subset = gate_subset,
                                               varm_info = components,
                                               dimred = "pca",
                                               dimred_key = dimred_key)
    
    adata = _merge_dimred_coordinates_into_adata(adata = adata,
                                                 gate_subset = gate_subset,
                                                 coordinates = coordinates,
                                                 dimred = "pca",
                                                 dimred_key = dimred_key)
    uns_dict = {"params": settings, **variances}
    adata = _add_uns_data(adata,
                          data = uns_dict,
                          key_added = dimred_key)

    return adata

def _merge_dimred_varm_info_into_adata(adata: AnnData,
                                       gate_subset: AnnData,
                                       varm_info: np.ndarray,
                                       dimred: Literal["umap", "tsne", "pca", "diffmap"],
                                       dimred_key: str) -> AnnData:
    ## to accomplish the merge, we first build a var dataframe 
    ## from the subsetted adata with the corresponding components.
    ## Merging the var-dataframes from the original and subsetted anndata
    ## gives the matching components array to be able to be merged
    ## into the original anndata

    varm_info_columns = [f"{dimred}{i}" for i in range(1, varm_info.shape[1] + 1)]
    gate_subset_var_frame = gate_subset.var.copy()
    adata_var_frame = adata.var.copy()
    gate_subset_var_frame = _add_array_into_var(gate_subset_var_frame,
                                                varm_info,
                                                colnames = varm_info_columns)
    combined_var_frame = _merge_var_frames(adata_var_frame = adata_var_frame,
                                           gate_subset_var_frame = gate_subset_var_frame)
    adata = _add_array_to_adata_slot(adata = adata,
                                     data = combined_var_frame,
                                     colnames = varm_info_columns,
                                     slot = "varm",
                                     key_added = dimred_key)
    return adata

def _merge_dimred_coordinates_into_adata(adata: AnnData,
                                         gate_subset: AnnData,
                                         coordinates: np.ndarray,
                                         dimred: Literal["umap", "tsne", "pca", "diffmap"],
                                         dimred_key: str) -> AnnData:
    coordinates_columns = [f"{dimred}{i}" for i in range(1, coordinates.shape[1] + 1)]

    gate_subset_obs_frame = gate_subset.obs.copy()
    adata_obs_frame = adata.obs.copy()
    gate_subset_obs_frame = _add_array_into_obs(gate_subset_obs_frame,
                                                coordinates,
                                                colnames = coordinates_columns)
    combined_obs_frame = _merge_obs_frames(adata_obs_frame = adata_obs_frame,
                                           gate_subset_obs_frame = gate_subset_obs_frame)
    adata = _add_array_to_adata_slot(adata,
                                     combined_obs_frame,
                                     colnames = coordinates_columns,
                                     slot = "obsm",
                                     key_added = f"X_{dimred_key}")
    
    return adata

def _choose_representation(adata: AnnData,
                           uns_key: str,
                           use_rep = None,
                           n_pcs = None):

    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = 'X'
    if use_rep is None:
        if adata.n_vars > 20:
            if f'X_pca_{uns_key}' in adata.obsm.keys():
                if n_pcs is not None and n_pcs > adata.obsm[f'X_pca_{uns_key}'].shape[1]:
                    raise ValueError(
                        '`X_pca` does not have enough PCs. Rerun `fp.tl.pca` with adjusted `n_comps`.'
                    )
                X = adata.obsm[f'X_pca_{uns_key}'][:, :n_pcs]
            else:
                raise ValueError(
                    'Please run `fp.tl.pca` for this gate first. ' 
                )
        else:
            X = adata.X
    else:
        if use_rep in adata.obsm.keys() and n_pcs is not None:
            if n_pcs > adata.obsm[use_rep].shape[1]:
                raise ValueError(
                    f'{use_rep} does not have enough Dimensions. Provide a '
                    'Representation with equal or more dimensions than'
                    '`n_pcs` or lower `n_pcs` '
                )
            X = adata.obsm[use_rep][:, :n_pcs]
        elif use_rep in adata.obsm.keys() and n_pcs is None:
            X = adata.obsm[use_rep]
        elif use_rep == 'X':
            X = adata.X
        else:
            raise ValueError(
                'Did not find {} in `.obsm.keys()`. '
                'You need to compute it first.'.format(use_rep)
            )
    return X

def _extract_valid_pca_kwargs(kwargs: dict) -> dict:
    valid_kwargs = ["n_comps", "zero_center", "svd_solver",
                    "random_state", "chunk", "chunk_size",
                    "whiten", "tol", "iterated_power",
                    "n_oversamples", "power_iteration_normalizer"]
    return {k: v for (k, v) in kwargs.items()
            if k in valid_kwargs}

def _extract_valid_neighbors_kwargs(kwargs: dict) -> dict:
    valid_kwargs = ["n_neighbors", "n_pcs", "use_rep",
                    "knn", "random_state", "method",
                    "metric", "metric_kwds", "key_added"]
    return {k: v for (k, v) in kwargs.items()
            if k in valid_kwargs}

def _extract_valid_tsne_kwargs(kwargs: dict) -> dict:
    valid_kwargs = ["n_components", "n_pcs", "use_rep",
                    "perplexity", "early_exaggeration",
                    "learning_rate", "random_state",
                    "use_fast_tsne", "n_jobs", "metric",
                    "n_iter", "n_iter_without_progress",
                    "min_grad_norm", "metric_params",
                    "init", "verbose", "method", "angle"]
    return {k: v for (k, v) in kwargs.items()
            if k in valid_kwargs}

def _extract_valid_umap_kwargs(kwargs: dict) -> dict:
    valid_kwargs = ["min_dist", "spread", "n_components",
                    "maxiter", "alpha", "gamma",
                    "negative_sample_rate", "init_pos",
                    "random_state", "a", "b", "method",
                    "neighbors_key"]
    return {k: v for (k, v) in kwargs.items()
            if k in valid_kwargs}

def _save_samplewise_dr_settings(adata: AnnData,
                                 data_group,
                                 data_metric,
                                 layer,
                                 use_only_fluo,
                                 exclude,
                                 scaling,
                                 reduction,
                                 n_components,
                                 **kwargs) -> None:
    if "settings" not in adata.uns:
        adata.uns["settings"] = {}

    settings_dict = {
        "data_group": data_group,
        "data_metric": data_metric,
        "layer": layer,
        "use_only_fluo": use_only_fluo,
        "exclude": exclude,
        "scaling": scaling,
        "n_components": n_components
    }
    settings_dict = {**settings_dict, **kwargs}
    adata.uns["settings"][f"_{reduction}_samplewise_{data_metric}_{layer}"] = settings_dict
    
    return

def _warn_user_about_changed_setting(dimred: str,
                                     parameter: str,
                                     new_value: str,
                                     reason: str) -> None:
    warning_kwargs = {
        "dimred": dimred,
        "parameter": parameter,
        "new_value": new_value,
        "reason": reason
    }
    warnings.warn(DimredSettingModificationWarning._construct_message(**warning_kwargs),
                  DimredSettingModificationWarning)
    return

def _warn_user_about_insufficient_sample_size(gate: str,
                                              n_samples_per_gate,
                                              n_components) -> None:
    warnings_params = {
        "gate": gate,
        "n_samples_per_gate": n_samples_per_gate,
        "n_components": n_components
    }
    warnings.warn(InsufficientSampleNumberWarning._construct_message(**warnings_params),
                  InsufficientSampleNumberWarning)
    return

def _choose_use_rep_as_scanpy(adata: AnnData,
                              uns_key: str,
                              use_rep: Optional[str],
                              n_pcs: Optional[int]) -> str:
    """Function that copies the internal sc.tl._utils._choose_representation.
    We have to copy it to avoid copy pasting the Neighbors Class
    to find out which representation scanpy would choose. The chosen
    representation is then adapted to FACSPy and passed as a kwarg"""
    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = 'X'
    if use_rep is None:
        if adata.n_vars > 20:
            if f'X_pca_{uns_key}' in adata.obsm.keys():
                if n_pcs is not None and n_pcs > adata.obsm[f'X_pca_{uns_key}'].shape[1]:
                    raise ValueError(
                        '`X_pca` does not have enough PCs. Rerun `fp.tl.pca` with adjusted `n_comps`.'
                    )
                return f"X_pca_{uns_key}"
            else:
                raise ValueError(
                    'Please run `fp.tl.pca` for this gate first. ' 
                )
        else:
            return "X"
    else:
        if use_rep in adata.obsm.keys() and n_pcs is not None:
            if n_pcs > adata.obsm[use_rep].shape[1]:
                raise ValueError(
                    f'{use_rep} does not have enough Dimensions. Provide a '
                    'Representation with equal or more dimensions than'
                    '`n_pcs` or lower `n_pcs` '
                )
            return use_rep
        elif use_rep in adata.obsm.keys() and n_pcs is None:
            return use_rep
        elif use_rep == 'X':
            return "X"
        else:
            raise ValueError(
                'Did not find {} in `.obsm.keys()`. '
                'You need to compute it first.'.format(use_rep)
            )


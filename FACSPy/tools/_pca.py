
from anndata import AnnData
from typing import Optional
import numpy as np
from typing import Union, Literal
from scipy.sparse._base import issparse

from ._utils import _merge_pca_info_into_adata
from ._dr_samplewise import _perform_samplewise_dr

def pca_samplewise(adata: AnnData,
                   data_group: Optional[Union[str, list[str]]] = "sample_ID",
                   data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                   layer: Literal["compensated", "transformed"] = "compensated",
                   use_only_fluo: bool = True,
                   exclude: Optional[Union[str, list, str]] = None,
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",
                   n_components: int = 3,
                   copy = False,
                   *args,
                   **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    adata = _perform_samplewise_dr(adata = adata,
                                   reduction = "PCA",
                                   data_metric = data_metric,
                                   data_group = data_group,
                                   layer = layer,
                                   use_only_fluo = use_only_fluo,
                                   exclude = exclude,
                                   scaling = scaling,
                                   n_components = n_components,
                                   *args,
                                   **kwargs)

    return adata if copy else None

def _pca(adata: AnnData,
         preprocessed_adata: AnnData,
         dimred_key: str,
         **kwargs) -> AnnData:
    """internal pca function to handle preprocessed datasets"""
    (coordinates,
     components,
     settings,
     variances) = _compute_pca(adata = preprocessed_adata,
                               **kwargs)
    
    adata = _merge_pca_info_into_adata(adata,
                                       preprocessed_adata,
                                       coordinates,
                                       components,
                                       settings,
                                       variances,
                                       dimred_key = dimred_key)
    
    return adata

def _compute_pca(adata: AnnData,
                 n_comps: Optional[int] = None,
                 zero_center: Optional[bool] = True,
                 svd_solver: str = "arpack",
                 random_state: int = 187,
                 chunked: bool = False,
                 chunk_size: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    
    # This module copies the sc.pp.pca function
    # with the important difference that nothing 
    # gets written to the dataset directly. That way,
    # we keep the anndatas as Views when multiple gates are
    # analyzed
    
    if n_comps is None:
        n_comps = min(adata.n_vars, adata.n_obs) - 1
    
    if chunked:
        from sklearn.decomposition import IncrementalPCA
        X_pca = np.zeros((adata.X.shape[0], n_comps), adata.X.dtype)

        pca_ = IncrementalPCA(n_components = n_comps)

        for chunk, _, _ in adata.chunked_X(chunk_size):
            chunk = chunk.toarray() if issparse(chunk) else chunk
            pca_.partial_fit(chunk)

        for chunk, start, end in adata.chunked_X(chunk_size):
            chunk = chunk.toarray() if issparse(chunk) else chunk
            X_pca[start:end] = pca_.transform(chunk)
    
    elif (not issparse(adata.X) or svd_solver == "randomized") and zero_center:
        from sklearn.decomposition import PCA

        if issparse(adata.X) and svd_solver == "randomized":
            adata.X = adata.X.toarray()
        
        pca_ = PCA(n_components = n_comps,
                   svd_solver = svd_solver,
                   random_state = random_state)
        X_pca = pca_.fit_transform(adata.X)

    elif issparse(adata.X) and zero_center:
        from sklearn.decomposition import PCA

        if svd_solver == "auto":
            svd_solver = "arpack"
        if svd_solver not in {"lobpcg", "arpack"}:
            raise ValueError("Cannot be used with sparse, use arpack or lobpcg instead")
        from scanpy.preprocessing._pca import _pca_with_sparse
        output = _pca_with_sparse(adata.X,
                                  n_comps,
                                  solver = svd_solver,
                                  random_state = random_state)
        
        X_pca = output["X_pca"]
        pca_ = PCA(n_components = n_comps,
                   svd_solver = svd_solver)
        pca_.components_ = output["components"]
        pca_.explained_variance_ = output["variance"]
        pca_.explained_variance_ratio_ = output["variance_ratio"]

    elif not zero_center:
        from sklearn.decomposition import TruncatedSVD

        pca_ = TruncatedSVD(n_components = n_comps,
                            random_state = random_state,
                            algorithm = svd_solver)
        X_pca = pca_.fit_transform(adata.X)
    
    else:
        raise Exception("This shouldnt happen")
    pca_settings = {
        "zero_center": zero_center
    }
    pca_variances = {
        "variance": pca_.explained_variance_,
        "variance_ratio": pca_.explained_variance_ratio_
    }
    assert adata.is_view
    return (
        X_pca, # pca_coordinates
        pca_.components_.T, # varm data
        pca_settings,
        pca_variances
    )
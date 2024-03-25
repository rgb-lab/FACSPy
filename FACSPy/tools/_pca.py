
from anndata import AnnData
from typing import Optional
import numpy as np
from typing import Union, Literal
from scipy.sparse._base import issparse

from ._utils import _merge_pca_info_into_adata
from ._dr_samplewise import _perform_samplewise_dr
from .._utils import _default_layer

@_default_layer
def pca_samplewise(adata: AnnData,
                   layer: str,
                   data_group: str = "sample_ID",
                   data_metric: Literal["mfi", "fop"] = "mfi",
                   use_only_fluo: bool = True,
                   exclude: Optional[Union[list[str], str]] = None,
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",
                   n_components: int = 3,
                   copy: bool = False,
                   *args,
                   **kwargs) -> Optional[AnnData]:
    """\
    Computes samplewise PCA based on either the median fluorescence values (MFI)
    or frequency of parent values (FOP). PCA will be calculated for all gates at once.
    The values are added to the corresponding `.uns` slot where MFI/FOP values are
    stored.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    n_components
        The number of components to be calculated. Defaults to 3.
    use_only_fluo
        Parameter to specify if the PCA should only be calculated for the fluorescence
        channels.
    exclude
        Can be used to exclude channels from calculating the embedding.
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score). Defaults to None.
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    copy
        Return a copy of adata instead of modifying inplace.
    **kwargs : dict, optional
        keyword arguments that are passed directly to the `sklearn.PCA`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.uns[f'{data_metric}_{data_group}_{layer}']`
            PCA coordinates are added to the respective frame
        `.uns['settings'][f"_pca_samplewise_{data_metric}_{layer}"]`
            Settings that were used for samplewise PCA calculation

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn', 'cofactors'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.settings.default_gate = "T_cells"
    >>> fp.settings.default_layer = "transformed"
    >>> fp.tl.mfi(dataset)
    >>> fp.tl.pca_samplewise(dataset)

    """

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
    """\
    Internal function to compute the PCA embedding. The core of the function
    is implemented from scanpy with the important difference that the PCA
    coordinates are returned and not written to the adata object.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    n_comps
        Number of components to be calculated.
    zero_center
        If `True`, compute standard PCA from covariance matrix.
        If `False`, omit zero-centering variables
        (uses :class:`~sklearn.decomposition.TruncatedSVD`),
        which allows to handle sparse input efficiently.
        Passing `None` decides automatically based on sparseness of the data.
    svd_solver
        One of `auto`, `full`, `arpack`, `randomized`
    random_state
        Sets the random state.
    chunked
        If `True`, perform an incremental PCA on segments of `chunk_size`.
        The incremental PCA automatically zero centers and ignores settings of
        `random_seed` and `svd_solver`. If `False`, perform a full PCA.
    chunk_size
        Number of observations to include in each chunk.
        Required if `chunked=True` was passed.

    Returns
    -------
    X_pca
        The PCA coordinates
    pca_.components
        Principal components containing the loadings
    pca_variances
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix
    pca_settings
        A dictionary containing the parameters used for analysis.

    """
    
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

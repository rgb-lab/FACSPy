from anndata import AnnData
from typing import Literal, Optional

from ._tsne import _tsne
from ._pca import _pca
from ._diffmap import _diffmap
from ._umap import _umap
from ._neighbors import _neighbors
from ._utils import (_preprocess_adata,
                     _extract_valid_pca_kwargs,
                     _extract_valid_neighbors_kwargs,
                     _extract_valid_tsne_kwargs,
                     _extract_valid_umap_kwargs,
                     _save_dr_settings,
                     _choose_use_rep_as_scanpy,
                     _recreate_preprocessed_view)

from .._utils import (_default_gate_and_default_layer,
                      _enable_gate_aliases,
                      IMPLEMENTED_SCALERS)
from ..exceptions._exceptions import InvalidScalingError

@_default_gate_and_default_layer
@_enable_gate_aliases
def pca(adata: AnnData,
        gate: str = None,
        layer: str = None,
        use_only_fluo: bool = True,
        exclude: Optional[list[str]] = None,
        scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
        copy: bool = False,
        **kwargs) -> Optional[AnnData]:
    """\
    Principal component analysis

    Computes PCA coordinates, loadings and variance decomposition

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where Rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    use_only_fluo
        Boolean parameter that controls whether only channels corresponding
        to fluorescence values are analyzed or if scatter (flow) and technical
        channels (CyTOF) are used for analysis. Defaults to True
    exclude
        gives the user the opportunity to exclude channels from the
        analysis. Lists or strings are allowed.
    scaling
        parameter that allows for scaling of the data matrix before
        PCA computation. Allowed values are "MinMaxScaler",
        "StandardScaler" or "RobustScaler".
    copy
        Return a copy of adata instead of modifying inplace
    **kwargs : dict, optional
        keyword arguments that are passed directly to the `_compute_pca`
        function. Please refer to its documentation.
    
    Returns
    -------
    adata : anndata.AnnData
        Returned if `copy = True`, otherwise adds fields to the anndata
        object:

        `.obsm['X_pca_{gate}_{layer}]`
            PCA representation of the data
        `.varm['PCs_{gate}_{layer}]`
            Principal components containing the loadings
        `.uns['pca_{gate}_{layer}]['variance']`
            Explained variance, equivalent to the eigenvalues of the
            covariance matrix
        `.uns['pca_{gate}_{layer}]['variance_ratio']`
            Ratio of explained variance.
        `.uns['settings']['_pca_{gate}_{layer}]`
            Settings that were used for PCA calculation
    """

    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      **kwargs)
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"pca_{uns_key}"

    adata = _pca(adata = adata,
                 preprocessed_adata = preprocessed_adata,
                 dimred_key = dimred_key,
                 **kwargs)    

    del adata.X
    return adata if copy else None

@_default_gate_and_default_layer
@_enable_gate_aliases
def diffmap(adata: AnnData,
            gate: str = None,
            layer: str = None,
            recalculate_pca: bool = False,
            use_only_fluo: bool = True,
            exclude: Optional[list[str]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            copy: bool = False,
            **kwargs) -> Optional[AnnData]:
    """\
    Diffusion Map Embedding calculation

    If PCA and neighbors have not been calculated for this gate and layer,
    the function will compute it automatically.

    From the scanpy docs:
    Diffusion maps [Coifman05]_ has been proposed for visualizing single-cell
    data by [Haghverdi15]_. The tool uses the adapted Gaussian kernel suggested
    by [Haghverdi16]_ in the implementation of [Wolf18]_.

    The width ("sigma") of the connectivity kernel is implicitly determined by
    the number of neighbors used to compute the single-cell graph in
    :func:`~scanpy.pp.neighbors`. To reproduce the original implementation
    using a Gaussian kernel, use `method=='gauss'` in
    :func:`~scanpy.pp.neighbors`. To use an exponential kernel, use the default
    `method=='umap'`. Differences between these options shouldn't usually be
    dramatic.

    Due to the Note in the scanpy documentation, that the 0th column is the steady
    state solution and not informative in diffusion maps, we discard the 0th
    position.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where Rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    use_only_fluo
        Boolean parameter that controls whether only channels corresponding
        to fluorescence values are analyzed or if scatter (flow) and technical
        channels (CyTOF) are used for analysis. Defaults to True
    exclude
        gives the user the opportunity to exclude channels from the
        analysis. Lists or strings are allowed.
    scaling
        parameter that allows for scaling of the data matrix before
        PCA computation. Allowed values are "MinMaxScaler",
        "StandardScaler" or "RobustScaler".
    copy
        Return a copy of adata instead of modifying inplace
    **kwargs : dict, optional
        keyword arguments that are passed directly to the `_compute_diffmap`
        function. Please refer to its documentation.
    
    Returns
    -------
    adata : anndata.AnnData
        Returned if `copy = True`, otherwise adds fields to the anndata
        object:

        `.obsm['X_diffmap_{gate}_{layer}]`
            DiffusionMap embedding of the data
        `.uns['diffmap_evals'_{gate}_{layer}]`
            Array of size (number of eigen vectors).
            Eigenvalues of transition matrix
        `.uns['settings']['_pca_{gate}_{layer}]`
            Settings that were used for PCA calculation
    """


    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      **kwargs)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"diffmap_{uns_key}"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    
    if f"X_pca_{uns_key}" not in adata.obsm or recalculate_pca:
        print("computing PCA for diffmap")
        pca_kwargs = _extract_valid_pca_kwargs(kwargs)
        adata = _pca(adata = adata,
                     preprocessed_adata = preprocessed_adata,
                     dimred_key = f"pca_{uns_key}",
                     **pca_kwargs)
        preprocessed_adata = _recreate_preprocessed_view(adata,
                                                         preprocessed_adata)

    if connectivities_key not in adata.obsp:
        print("computing neighbors for diffmap")
        neighbors_kwargs = _extract_valid_neighbors_kwargs(kwargs)
        if not "use_rep" in neighbors_kwargs:
            neighbors_kwargs["use_rep"] = _choose_use_rep_as_scanpy(adata,
                                                                    uns_key = uns_key,
                                                                    use_rep = None,
                                                                    n_pcs = neighbors_kwargs.get("n_pcs"))
        adata = _neighbors(adata = adata,
                           preprocessed_adata = preprocessed_adata,
                           neighbors_key = neighbors_key,
                           **neighbors_kwargs)
        preprocessed_adata = _recreate_preprocessed_view(adata,
                                                         preprocessed_adata)

    adata = _diffmap(adata = adata,
                     preprocessed_adata = preprocessed_adata,
                     neighbors_key = neighbors_key,
                     uns_key = uns_key,
                     dimred_key = dimred_key,
                     **kwargs)
    del adata.X
    return adata if copy else None

@_default_gate_and_default_layer
@_enable_gate_aliases
def umap(adata: AnnData,
         gate: str = None,
         layer: str = None,
         recalculate_pca: bool = False,
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      *args,
                      **kwargs)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"umap_{uns_key}"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    if kwargs.get("use_rep") is None:
        if f"X_pca_{uns_key}" not in adata.obsm or recalculate_pca:
            pca_kwargs = _extract_valid_pca_kwargs(kwargs)
            adata = _pca(adata = adata,
                         preprocessed_adata = preprocessed_adata,
                         dimred_key = f"pca_{uns_key}",
                         **pca_kwargs)
            preprocessed_adata = _recreate_preprocessed_view(adata,
                                                             preprocessed_adata)
            print(adata)
            print(preprocessed_adata)
            assert "X_pca_live_compensated" in preprocessed_adata.obsm
            


    if connectivities_key not in adata.obsp:
        print("computing neighbors for umap")
        neighbors_kwargs = _extract_valid_neighbors_kwargs(kwargs)
        if not "use_rep" in neighbors_kwargs:
            neighbors_kwargs["use_rep"] = _choose_use_rep_as_scanpy(adata,
                                                                    uns_key = uns_key,
                                                                    use_rep = None,
                                                                    n_pcs = neighbors_kwargs.get("n_pcs"))
        adata = _neighbors(adata = adata,
                           preprocessed_adata = preprocessed_adata,
                           neighbors_key = neighbors_key,
                           **neighbors_kwargs)
        preprocessed_adata = _recreate_preprocessed_view(adata,
                                                         preprocessed_adata)
        print(preprocessed_adata)

    umap_kwargs = _extract_valid_umap_kwargs(kwargs)
    adata = _umap(adata = adata,
                  preprocessed_adata = preprocessed_adata,
                  neighbors_key = neighbors_key,
                  dimred_key = dimred_key,
                  uns_key = uns_key,
                  **umap_kwargs)

    del adata.X

    return adata if copy else None

@_default_gate_and_default_layer
@_enable_gate_aliases
def tsne(adata: AnnData,
         gate: str = None,
         layer: str = None,
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         recalculate_pca: bool = False,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)
    
    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      *args,
                      **kwargs)

    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"tsne_{uns_key}"
    
    if kwargs.get("use_rep") is None:
        if f"X_pca_{uns_key}" not in adata.obsm or recalculate_pca:
            pca_kwargs = _extract_valid_pca_kwargs(kwargs)
            adata = _pca(adata = adata,
                         preprocessed_adata = preprocessed_adata,
                         dimred_key = dimred_key,
                         **pca_kwargs)

    tsne_kwargs = _extract_valid_tsne_kwargs(kwargs)
    adata = _tsne(adata = adata,
                  preprocessed_adata = preprocessed_adata,
                  uns_key = uns_key,
                  dimred_key = dimred_key,
                  **tsne_kwargs)
    del adata.X
    return adata if copy else None


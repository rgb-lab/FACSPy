from anndata import AnnData
from typing import Optional, Literal, Union
from scipy.sparse import csr_matrix

from ._utils import (_preprocess_adata,
                     _merge_neighbors_info_into_adata,
                     _choose_use_rep_as_scanpy)
from .._utils import _default_gate_and_default_layer, _enable_gate_aliases

@_default_gate_and_default_layer
@_enable_gate_aliases
def neighbors(adata: AnnData,
              gate: str,
              layer: str,
              use_only_fluo: bool = True,
              exclude: Optional[Union[list[str], str]] = None,
              scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
              n_neighbors: int = 15,
              use_rep: Optional[str] = None,
              n_pcs: Optional[int] = None,
              copy: bool = False,
              *args,
              **kwargs) -> Optional[AnnData]:
    """\
    Computes Neighbors. 

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    use_only_fluo
        Parameter to specify if the UMAP should only be calculated for the fluorescence
        channels. Specify `recalculate_pca` to repeat PCA calculation.
    exclude
        Can be used to exclude channels from calculating the embedding.
        Specify `recalculate_pca` to repeat PCA calculation.
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score). Defaults to None.
    n_neighbors
        Number of neighbors to compute.
    use_rep
        Representation to compute neighbors on. Defaults to `X_pca_{gate}_{layer}`
    n_pcs
        Number of principal components to use.
    copy
        Return a copy of adata instead of modifying inplace.
    **kwargs : dict, optional
        keyword arguments that are passed directly to the `_compute_neighbors`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.obsm[f'{gate}_{layer}_neighbors_connectivities']`
            connectivities
        `.obsm[f'{gate}_{layer}_neighbors_distances']`
            distances

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
    >>> fp.tl.pca(dataset)
    >>> fp.tl.neighbors(dataset)

    """

    adata = adata.copy() if copy else adata

    if exclude is None:
        exclude = []
    else:
        if not isinstance(exclude, list):
            exclude = [exclude]    

    uns_key = f"{gate}_{layer}"
    neighbors_key = f"{uns_key}_neighbors"

    if use_rep is None:
        use_rep = _choose_use_rep_as_scanpy(adata,
                                            uns_key = uns_key,
                                            use_rep = use_rep,
                                            n_pcs = n_pcs)
 
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    adata = _neighbors(adata = adata,
                       preprocessed_adata = preprocessed_adata,
                       neighbors_key = neighbors_key,
                       use_rep = use_rep,
                       n_pcs = n_pcs,
                       n_neighbors = n_neighbors,
                       *args,
                       **kwargs)
    return adata if copy else None

def _neighbors(adata: AnnData,
               preprocessed_adata: AnnData,
               neighbors_key: str,
               **kwargs) -> None:
    """internal neighbors function which can handle preprocessed adatas"""
    (distances,
     connectivities,
     neighbors_dict) = _compute_neighbors(preprocessed_adata,
                                          key_added = neighbors_key,
                                          **kwargs)

    adata = _merge_neighbors_info_into_adata(adata,
                                             preprocessed_adata,
                                             connectivities = connectivities,
                                             distances = distances,
                                             neighbors_dict = neighbors_dict,
                                             neighbors_key = neighbors_key)
    return adata

def _compute_neighbors(adata: AnnData,
                       n_neighbors: int = 15,
                       n_pcs: Optional[int] = None,
                       use_rep: Optional[str] = None,
                       knn: bool = True,
                       random_state: int = 187,
                       method: Optional[str] = "umap",
                       metric: str = "euclidean",
                       metric_kwds: dict = None,
                       key_added: Optional[str] = None) -> tuple[csr_matrix, csr_matrix]:
    # This module copies the sc.pp.neighbors function
    # with the important difference that nothing 
    # gets written to the dataset directly. That way,
    # we keep the anndatas as Views when multiple gates are
    # analyzed
    from scanpy.neighbors import Neighbors
    if metric_kwds is None:
        metric_kwds = {}

    neighbors = Neighbors(adata)
    neighbors.compute_neighbors(
        n_neighbors = n_neighbors,
        knn = knn,
        n_pcs = n_pcs,
        use_rep = use_rep,
        method = method,
        metric = metric,
        metric_kwds = metric_kwds,
        random_state = random_state,
    )

    if key_added is None:
        key_added = 'neighbors'
        conns_key = 'connectivities'
        dists_key = 'distances'
    else:
        conns_key = key_added + '_connectivities'
        dists_key = key_added + '_distances'

    neighbors_dict = {}
    neighbors_dict['connectivities_key'] = conns_key
    neighbors_dict['distances_key'] = dists_key

    neighbors_dict['params'] = {'n_neighbors': neighbors.n_neighbors, 'method': method}
    neighbors_dict['params']['random_state'] = random_state
    neighbors_dict['params']['metric'] = metric
    if metric_kwds:
        neighbors_dict['params']['metric_kwds'] = metric_kwds
    if use_rep is not None:
        neighbors_dict['params']['use_rep'] = use_rep
    if n_pcs is not None:
        neighbors_dict['params']['n_pcs'] = n_pcs
    if neighbors.rp_forest is not None:
        neighbors_dict['rp_forest'] = neighbors.rp_forest

    return (neighbors.distances, neighbors.connectivities, neighbors_dict)

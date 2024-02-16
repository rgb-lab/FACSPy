from anndata import AnnData
from typing import Optional, Literal
from scipy.sparse import csr_matrix

from ._utils import (_preprocess_adata,
                     _merge_neighbors_info_into_adata,
                     _choose_use_rep_as_scanpy)
from .._utils import _default_gate_and_default_layer, _enable_gate_aliases

@_default_gate_and_default_layer
@_enable_gate_aliases
def neighbors(adata: AnnData,
              gate: str = None,
              layer: str = None,
              use_only_fluo: bool = True,
              exclude: Optional[list[str]] = None,
              scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
              n_neighbors: int = 15,
              use_rep: str = None,
              n_pcs: int = None,
              copy: bool = False,
              *args,
              **kwargs) -> Optional[AnnData]:
    """Function to add neighbors to adata"""
    adata = adata.copy() if copy else adata
    
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
 
 


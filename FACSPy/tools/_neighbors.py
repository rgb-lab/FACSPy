from anndata import AnnData
from typing import Optional
from scipy.sparse import csr_matrix

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
 
 


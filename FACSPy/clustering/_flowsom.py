from typing import Optional, Literal

from anndata import AnnData
from typing import Union
from ..utils import contains_only_fluo

from FlowSOM.cluster import flowsom

def flowsom_cluster(adata: AnnData,
                    on: Literal["compensated", "transformed"] = "transformed",
                    exclude: Optional[Union[str, list[str]]] = None,
                    copy: bool = False,
                    x_dim: int = 50,
                    y_dim: int = 50,
                    sigma: float = 1,
                    learning_rate: float = 0.5,
                    n_iterations: int = 100,
                    neighborhood_function = "gaussian",
                    consensus_cluster_algorithm: str = "AgglomerativeClustering",
                    consensus_cluster_min_n: int = 10,
                    consensus_cluster_max_n: int = 50,
                    consensus_cluster_resample_proportion: float = 0.5,
                    consensus_cluster_n_resamples: int = 10,
                    verbose: bool = False,
                    n_jobs: int = None,
                    random_state: int = 187) -> Optional[AnnData]:
    
    if parc_kwargs is None:
        parc_kwargs = {}
    cluster_set = adata.copy() if copy else adata

    assert contains_only_fluo(cluster_set)

    if exclude:
        cluster_set = adata[:, [var for var in adata.var_names if var not in exclude]]
        assert adata.isview

    ### we take the raw data as they are mostly below 50 markers anyway and would probably be not too much higher
    adata.obs["flowsom_labels"] = flowsom(adata.layers[on],
                                          x_dim = x_dim,
                                          y_dim = y_dim,
                                          sigma = sigma,
                                          learning_rate = learning_rate,
                                          n_iterations = n_iterations,
                                          neighborhood_function = neighborhood_function,
                                          consensus_cluster_algorithm = consensus_cluster_algorithm,
                                          consensus_cluster_min_n = consensus_cluster_min_n,
                                          consensus_cluster_max_n = consensus_cluster_max_n,
                                          consensus_cluster_n_resamples = consensus_cluster_n_resamples,
                                          consensus_cluster_resample_proportion = consensus_cluster_resample_proportion,
                                          verbose = verbose,
                                          random_state = random_state,
                                          n_jobs = n_jobs)
    adata.obs["flowsom_labels"] = adata.obs["flowsom_labels"].astype("category")

    return adata if copy else None




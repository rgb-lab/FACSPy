from typing import Optional, Literal, Union

import phenograph
import anndata as ad
from ..utils import contains_only_fluo

def phenograph_cluster(adata: ad.AnnData,
                       key_added: str = "phenograph",
                       on: Literal["compensated", "transformed"] = "transformed",
                       algorithm: Literal["leiden", "louvain"] = "leiden",
                       exclude: Optional[Union[str, list[str]]] = None,
                       copy: bool = False,
                       cluster_kwargs: dict = None) -> Optional[ad.AnnData]:

    if cluster_kwargs is None:
        cluster_kwargs = {}
    cluster_set = adata.copy() if copy else adata

    assert contains_only_fluo(cluster_set)

    if exclude:
        cluster_set = adata[:, [var for var in adata.var_names if var not in exclude]]
        assert adata.isview

    communities, graph, Q = phenograph.cluster(cluster_set.layers[on],
                                               clustering_algo = algorithm,
                                               **cluster_kwargs)

    adata.obs[f"{key_added}_{algorithm}"] = communities
    adata.obs[f"{key_added}_{algorithm}"] = adata.obs[f"{key_added}_{algorithm}"].astype("category")
    adata.uns[f"{key_added}_{algorithm}_graph"] = graph
    adata.uns[f"{key_added}_{algorithm}_Q"] = Q

    return adata if copy else None
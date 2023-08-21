from typing import Optional, Literal

from anndata import AnnData
from typing import Union
from ..utils import contains_only_fluo

from FlowSOM.cluster import flowsom

def flowsom_cluster(adata: AnnData,
                    on: Literal["compensated", "transformed"] = "transformed",
                    exclude: Optional[Union[str, list[str]]] = None,
                    copy: bool = False,
                    cluster_kwargs: dict = None) -> Optional[AnnData]:
    if cluster_kwargs is None:
        from multiprocessing import cpu_count
        cluster_kwargs = {
            "x_dim": 50,
            "y_dim": 50,
            "n_jobs": cpu_count() - 2
        }
    cluster_set = adata.copy() if copy else adata

    assert contains_only_fluo(cluster_set)

    if "consensus_cluster_max_n" not in cluster_kwargs:
        cluster_kwargs["consensus_cluster_max_n"] = min(50, adata.shape[0])

    if exclude:
        cluster_set = adata[:, [var for var in adata.var_names if var not in exclude]]
        assert adata.isview

    ### we take the raw data as they are mostly below 50 markers anyway and would probably be not too much higher
    adata.obs["flowsom_labels"] = flowsom(cluster_set.layers[on],
                                          **cluster_kwargs)
    adata.obs["flowsom_labels"] = adata.obs["flowsom_labels"].astype("category")

    return adata if copy else None




 
from ..utils import subset_fluo_channels

import scanpy as sc
from anndata import AnnData

from typing import Optional, Union, Literal

from ..utils import contains_only_fluo


def leiden_cluster(adata: AnnData,
                   on: Literal["compensated", "transformed"] = "transformed",
                   exclude: Optional[Union[str, list[str]]] = None,
                   copy: bool = False,
                   leiden_kwargs=None) -> Optional[AnnData]:
    
    if leiden_kwargs is None:
        leiden_kwargs = {}
    cluster_set = adata.copy() if copy else adata

    assert contains_only_fluo(cluster_set)

    if exclude:
        cluster_set = adata[:, [var for var in adata.var_names if var not in exclude]]
        assert adata.isview

    ### we take the raw data as they are mostly below 50 markers anyway and would probably be not too much higher
    sc.tl.leiden(cluster_set,
                 **leiden_kwargs)

    adata.obs["leiden"] = cluster_set.obs["leiden"]

    return adata if copy else None
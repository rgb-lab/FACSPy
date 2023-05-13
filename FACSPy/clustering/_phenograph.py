from typing import Optional, Literal

import phenograph
import anndata as ad
from ..utils import subset_fluo_channels

def phenograph(dataset: ad.AnnData,
               key_added: str = "phenograph",
               algorithm: Literal["leiden", "louvain"] = "leiden",
               copy: bool = False) -> Optional[ad.AnnData]:
    
    fluo_dataset = subset_fluo_channels(dataset)
    communities, graph, Q = phenograph.cluster(fluo_dataset.layers["transformed"],
                                               clustering_algo = algorithm)

    dataset.obs[f"{key_added}_{algorithm}"] = communities
    dataset.uns[f"{key_added}_{algorithm}_graph"] = graph
    dataset.uns[f"{key_added}_{algorithm}_Q"] = Q

    return dataset if copy else None
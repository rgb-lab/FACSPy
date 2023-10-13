import phenograph as _phenograph
from anndata import AnnData

from typing import Optional, Literal, Union

from .utils import preprocess_adata, merge_cluster_info_into_adata

def phenograph(adata: AnnData,
               gate: str,
               data_origin: Literal["compensated", "transformed"] = "transformed",
               algorithm: Literal["leiden", "louvain"] = "leiden",
               use_only_fluo: bool = True,
               exclude: Optional[Union[str, list[str]]] = None,
               scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
               copy: bool = False,
               *args,
               **kwargs) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata

    uns_key = f"{gate}_{data_origin}"
    cluster_key = f"{uns_key}_phenograph"

    preprocessed_adata = preprocess_adata(adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)

    communities, graph, Q = _phenograph.cluster(preprocessed_adata.layers[data_origin],
                                                clustering_algo = algorithm,
                                                *args,
                                                **kwargs)

    adata = merge_cluster_info_into_adata(adata,
                                          preprocessed_adata,
                                          cluster_key = cluster_key,
                                          cluster_assignments = communities)

    adata.uns[f"{cluster_key}_graph"] = graph
    adata.uns[f"{cluster_key}_Q"] = Q

    return adata if copy else None
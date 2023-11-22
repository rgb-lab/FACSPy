import phenograph as _phenograph
from anndata import AnnData

from typing import Optional, Literal, Union

from ._utils import (_preprocess_adata,
                     _merge_cluster_info_into_adata,
                     _save_cluster_settings)
from .._utils import IMPLEMENTED_SCALERS
from ..exceptions._exceptions import InvalidScalingError


def phenograph(adata: AnnData,
               gate: str,
               layer: Literal["compensated", "transformed"] = "transformed",
               use_only_fluo: bool = True,
               exclude: Optional[Union[str, list[str]]] = None,
               scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
               copy: bool = False,
               **kwargs) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_cluster_settings(adata = adata,
                           gate = gate,
                           layer = layer,
                           use_only_fluo = use_only_fluo,
                           exclude = exclude,
                           scaling = scaling,
                           clustering = "phenograph",
                           **kwargs)

    uns_key = f"{gate}_{layer}"
    cluster_key = f"{uns_key}_phenograph"

    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    communities, graph, Q = _phenograph.cluster(preprocessed_adata.X,
                                                clustering_algo = kwargs.get("clustering_algo", "leiden"),
                                                **kwargs)

    adata = _merge_cluster_info_into_adata(adata,
                                           preprocessed_adata,
                                           cluster_key = cluster_key,
                                           cluster_assignments = communities)

    adata.uns[f"{cluster_key}_graph"] = graph
    adata.uns[f"{cluster_key}_Q"] = Q

    return adata if copy else None
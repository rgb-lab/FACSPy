
from anndata import AnnData
from FlowSOM import flowsom as _flowsom

from typing import Optional, Literal, Union

from ._utils import (preprocess_adata,
                     merge_cluster_info_into_adata)

def flowsom(adata: AnnData,
            gate: str,
            data_origin: Literal["compensated", "transformed"] = "transformed",
            use_only_fluo: bool = True,
            exclude: Optional[Union[str, list[str]]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            copy: bool = False,
            *args,
            **kwargs) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata

    uns_key = f"{gate}_{data_origin}"
    cluster_key = f"{uns_key}_flowsom"
    if not kwargs:
        from multiprocessing import cpu_count
        kwargs = {
            "x_dim": 50,
            "y_dim": 50,
            "n_jobs": cpu_count() - 2
        }
    if "consensus_cluster_max_n" not in kwargs:
        kwargs["consensus_cluster_max_n"] = min(50, adata.shape[0])

    preprocessed_adata = preprocess_adata(adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)
    cluster_annotations = _flowsom(preprocessed_adata.layers[data_origin],
                                   *args,
                                   **kwargs)
    adata = merge_cluster_info_into_adata(adata,
                                          preprocessed_adata,
                                          cluster_key = cluster_key,
                                          cluster_assignments = cluster_annotations)
    
    return adata if copy else None
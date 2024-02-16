
from anndata import AnnData
from FlowSOM import flowsom as _flowsom

from typing import Optional, Literal, Union

from ._utils import (_preprocess_adata,
                     _merge_cluster_info_into_adata,
                     _save_cluster_settings)

from .._utils import (_default_gate_and_default_layer,
                      _default_gate,
                      _default_layer,
                      _enable_gate_aliases,
                      IMPLEMENTED_SCALERS)
from ..exceptions._exceptions import InvalidScalingError

@_default_gate_and_default_layer
@_enable_gate_aliases
def flowsom(adata: AnnData,
            gate: str = None,
            layer: str = None,
            use_only_fluo: bool = True,
            exclude: Optional[Union[str, list[str]]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            key_added: Optional[str] = None,
            copy: bool = False,
            **kwargs) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    uns_key = f"{gate}_{layer}"
    cluster_key = key_added or f"{uns_key}_flowsom"

    if not kwargs:
        from multiprocessing import cpu_count
        kwargs = {
            "x_dim": 50,
            "y_dim": 50,
            "n_jobs": cpu_count() - 2
        }

    if "consensus_cluster_max_n" not in kwargs:
        kwargs["consensus_cluster_max_n"] = min(50, adata.shape[0])

    _save_cluster_settings(adata = adata,
                           gate = gate,
                           layer = layer,
                           use_only_fluo = use_only_fluo,
                           exclude = exclude,
                           scaling = scaling,
                           clustering = "flowsom",
                           **kwargs)

    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    cluster_annotations = _flowsom(preprocessed_adata.X,
                                   **kwargs)

    adata = _merge_cluster_info_into_adata(adata,
                                           preprocessed_adata,
                                           cluster_key = cluster_key,
                                           cluster_assignments = cluster_annotations)
    
    return adata if copy else None
import parc as _parc
from anndata import AnnData

from typing import Optional, Union, Literal

from ._utils import (_preprocess_adata,
                     _merge_cluster_info_into_adata,
                     _extract_valid_neighbors_kwargs,
                     _extract_valid_parc_kwargs,
                     _save_cluster_settings,
                     _choose_use_rep_as_scanpy,
                     _recreate_preprocessed_view)
from ._neighbors import _neighbors

from .._utils import (_default_gate_and_default_layer,
                      IMPLEMENTED_SCALERS)
from ..exceptions._exceptions import InvalidScalingError

@_default_gate_and_default_layer
def parc(adata: AnnData,
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

    _save_cluster_settings(adata = adata,
                           gate = gate,
                           layer = layer,
                           use_only_fluo = use_only_fluo,
                           exclude = exclude,
                           scaling = scaling,
                           clustering = "parc",
                           **kwargs)

    uns_key = f"{gate}_{layer}"
    cluster_key = key_added or f"{uns_key}_parc"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"

    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    if connectivities_key not in adata.obsp:
        print("computing neighbors for parc!")
        neighbors_kwargs = _extract_valid_neighbors_kwargs(kwargs)
        if not "use_rep" in neighbors_kwargs:
            neighbors_kwargs["use_rep"] = _choose_use_rep_as_scanpy(adata,
                                                                    uns_key = uns_key,
                                                                    use_rep = None,
                                                                    n_pcs = neighbors_kwargs.get("n_pcs"))
        adata = _neighbors(adata = adata,
                           preprocessed_adata = preprocessed_adata,
                           neighbors_key = neighbors_key,
                           **neighbors_kwargs)
        preprocessed_adata = _recreate_preprocessed_view(adata,
                                                         preprocessed_adata)

    parc_kwargs = _extract_valid_parc_kwargs(kwargs)
    parcer = _parc.PARC(preprocessed_adata.X,
                        neighbor_graph = adata.obsp[connectivities_key]
                                         if connectivities_key in adata.obsp else None,
                        **parc_kwargs)
    parcer.run_PARC()

    adata = _merge_cluster_info_into_adata(adata,
                                           preprocessed_adata,
                                           cluster_key = cluster_key,
                                           cluster_assignments = parcer.labels)
       
    return adata if copy else None
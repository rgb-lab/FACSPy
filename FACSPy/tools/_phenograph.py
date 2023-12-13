import phenograph as _phenograph
from anndata import AnnData

from typing import Optional, Literal, Union

from ._utils import (_preprocess_adata,
                     _merge_cluster_info_into_adata,
                     _save_cluster_settings,
                     _extract_valid_pca_kwargs,
                     _extract_valid_neighbors_kwargs,
                     _choose_use_rep_as_scanpy,
                     _recreate_preprocessed_view)
from ._pca import _pca
from ._neighbors import _neighbors
from .._utils import (_default_gate_and_default_layer,
                      IMPLEMENTED_SCALERS)
from ..exceptions._exceptions import InvalidScalingError

@_default_gate_and_default_layer
def phenograph(adata: AnnData,
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

    if not "clustering_algo" in kwargs:
        kwargs["clustering_algo"] = "leiden"

    _save_cluster_settings(adata = adata,
                           gate = gate,
                           layer = layer,
                           use_only_fluo = use_only_fluo,
                           exclude = exclude,
                           scaling = scaling,
                           clustering = "phenograph",
                           **kwargs)

    uns_key = f"{gate}_{layer}"
    cluster_key = key_added or f"{uns_key}_phenograph"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"

    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

#    if f"X_pca_{uns_key}" not in adata.obsm:
#        print("computing PCA for phenograph!")
#        pca_kwargs = _extract_valid_pca_kwargs(kwargs)
#        adata = _pca(adata = adata,
#                     preprocessed_adata = preprocessed_adata,
#                     dimred_key = f"pca_{uns_key}",
#                     **pca_kwargs)
#        preprocessed_adata = _recreate_preprocessed_view(adata,
#                                                         preprocessed_adata)
#
#    if connectivities_key not in adata.obsp:
#        print("computing neighbors for phenograph!")
#        neighbors_kwargs = _extract_valid_neighbors_kwargs(kwargs)
#        if not "use_rep" in neighbors_kwargs:
#            neighbors_kwargs["use_rep"] = _choose_use_rep_as_scanpy(adata,
#                                                                    uns_key = uns_key,
#                                                                    use_rep = None,
#                                                                    n_pcs = neighbors_kwargs.get("n_pcs"))
#        adata = _neighbors(adata = adata,
#                           preprocessed_adata = preprocessed_adata,
#                           neighbors_key = neighbors_key,
#                           **neighbors_kwargs)
#        preprocessed_adata = _recreate_preprocessed_view(adata,
#                                                         preprocessed_adata)


    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    if ("k" in kwargs and kwargs["k"] <= preprocessed_adata.shape[0]) or "k" not in kwargs:
        print(f"warning! Setting k to {min(preprocessed_adata.shape[0] - 1, 30)} to avoid errors")
        kwargs["k"] = min(preprocessed_adata.shape[0]-1, 30)

    communities, graph, Q = _phenograph.cluster(preprocessed_adata.X,
                                                **kwargs)

    adata = _merge_cluster_info_into_adata(adata,
                                           preprocessed_adata,
                                           cluster_key = cluster_key,
                                           cluster_assignments = communities)

    adata.uns[f"{cluster_key}_graph"] = graph
    adata.uns[f"{cluster_key}_Q"] = Q

    return adata if copy else None
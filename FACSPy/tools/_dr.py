from anndata import AnnData
from typing import Literal, Optional

from ._tsne import _tsne
from ._pca import _pca
from ._diffmap import _diffmap
from ._umap import _umap
from ._neighbors import _neighbors
from ._utils import (_preprocess_adata,
                     _extract_valid_pca_kwargs,
                     _extract_valid_neighbors_kwargs,
                     _extract_valid_tsne_kwargs,
                     _extract_valid_umap_kwargs,
                     _save_dr_settings,
                     _choose_use_rep_as_scanpy,
                     _recreate_preprocessed_view)

from .._utils import (_default_gate_and_default_layer,
                      IMPLEMENTED_SCALERS)
from ..exceptions._exceptions import InvalidScalingError

@_default_gate_and_default_layer
def pca(adata: AnnData,
        gate: str = None,
        layer: str = None,
        use_only_fluo: bool = True,
        exclude: Optional[list[str]] = None,
        scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
        copy: bool = False,
        **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      **kwargs)
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"pca_{uns_key}"

    adata = _pca(adata = adata,
                 preprocessed_adata = preprocessed_adata,
                 dimred_key = dimred_key,
                 **kwargs)    

    del adata.X
    return adata if copy else None

@_default_gate_and_default_layer
def diffmap(adata: AnnData,
            gate: str = None,
            layer: str = None,
            recalculate_pca: bool = False,
            use_only_fluo: bool = True,
            exclude: Optional[list[str]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            copy: bool = False,
            *args,
            **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      *args,
                      **kwargs)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"diffmap_{uns_key}"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    
    if f"X_pca_{uns_key}" not in adata.obsm or recalculate_pca:
        print("computing PCA for diffmap")
        pca_kwargs = _extract_valid_pca_kwargs(kwargs)
        adata = _pca(adata = adata,
                     preprocessed_adata = preprocessed_adata,
                     dimred_key = f"pca_{uns_key}",
                     **pca_kwargs)
        preprocessed_adata = _recreate_preprocessed_view(adata,
                                                         preprocessed_adata)

    if connectivities_key not in adata.obsp:
        print("computing neighbors for diffmap")
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

    adata = _diffmap(adata = adata,
                     preprocessed_adata = preprocessed_adata,
                     neighbors_key = neighbors_key,
                     uns_key = uns_key,
                     dimred_key = dimred_key,
                     **kwargs)
    del adata.X
    return adata if copy else None

@_default_gate_and_default_layer
def umap(adata: AnnData,
         gate: str = None,
         layer: str = None,
         recalculate_pca: bool = False,
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      *args,
                      **kwargs)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"umap_{uns_key}"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    if kwargs.get("use_rep") is None:
        if f"X_pca_{uns_key}" not in adata.obsm or recalculate_pca:
            pca_kwargs = _extract_valid_pca_kwargs(kwargs)
            adata = _pca(adata = adata,
                         preprocessed_adata = preprocessed_adata,
                         dimred_key = f"pca_{uns_key}",
                         **pca_kwargs)
            preprocessed_adata = _recreate_preprocessed_view(adata,
                                                             preprocessed_adata)
            print(adata)
            print(preprocessed_adata)
            assert "X_pca_live_compensated" in preprocessed_adata.obsm
            


    if connectivities_key not in adata.obsp:
        print("computing neighbors for umap")
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
        print(preprocessed_adata)

    umap_kwargs = _extract_valid_umap_kwargs(kwargs)
    adata = _umap(adata = adata,
                  preprocessed_adata = preprocessed_adata,
                  neighbors_key = neighbors_key,
                  dimred_key = dimred_key,
                  uns_key = uns_key,
                  **umap_kwargs)

    del adata.X

    return adata if copy else None

@_default_gate_and_default_layer
def tsne(adata: AnnData,
         gate: str = None,
         layer: str = None,
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         recalculate_pca: bool = False,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    exclude = [] if exclude is None else exclude

    if not scaling in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)
    
    _save_dr_settings(adata = adata,
                      gate = gate,
                      layer = layer,
                      use_only_fluo = use_only_fluo,
                      exclude = exclude,
                      scaling = scaling,
                      reduction = "pca",
                      *args,
                      **kwargs)

    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    
    uns_key = f"{gate}_{layer}"
    dimred_key = f"tsne_{uns_key}"
    
    if kwargs.get("use_rep") is None:
        if f"X_pca_{uns_key}" not in adata.obsm or recalculate_pca:
            pca_kwargs = _extract_valid_pca_kwargs(kwargs)
            adata = _pca(adata = adata,
                         preprocessed_adata = preprocessed_adata,
                         dimred_key = dimred_key,
                         **pca_kwargs)

    tsne_kwargs = _extract_valid_tsne_kwargs(kwargs)
    adata = _tsne(adata = adata,
                  preprocessed_adata = preprocessed_adata,
                  uns_key = uns_key,
                  dimred_key = dimred_key,
                  **tsne_kwargs)
    del adata.X
    return adata if copy else None


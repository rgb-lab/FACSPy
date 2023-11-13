import scanpy as sc

from anndata import AnnData
from typing import Literal, Optional

from ._utils import (_preprocess_adata,
                     merge_pca_info_into_adata,
                     merge_neighbors_info_into_adata,
                     merge_dimred_coordinates_into_adata,
                     add_uns_data)

def _save_dr_settings(adata: AnnData,
                      gate,
                      layer,
                      use_only_fluo,
                      exclude,
                      scaling,
                      reduction,
                      **kwargs) -> None:
    if "settings" not in adata.uns:
        adata.uns["settings"] = {}

    settings_dict = {
        "gate": gate,
        "layer": layer,
        "use_only_fluo": use_only_fluo,
        "exclude": exclude,
        "scaling": scaling,
    }
    settings_dict = {**settings_dict, **kwargs}
    adata.uns["settings"][f"_{reduction}_{layer}"] = settings_dict
    
    return


def pca(adata: AnnData,
        gate: str,
        layer: Literal["compensated", "transformed"] = "transformed",
        use_only_fluo: bool = True,
        exclude: Optional[list[str]] = None,
        scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
        copy: bool = False,
        *args,
        **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

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
    dimred_key = f"{uns_key}_pca"
    
    sc.pp.pca(preprocessed_adata,
              random_state = 187,
              *args,
              **kwargs)
    
    adata = merge_pca_info_into_adata(adata,
                                      preprocessed_adata,
                                      dimred_key = dimred_key,
                                      uns_key = uns_key)

    return adata if copy else None

def diffmap(adata: AnnData,
            gate: str,
            layer: Literal["compensated", "transformed"] = "transformed",
            recalculate_pca: bool = False,
            use_only_fluo: bool = True,
            exclude: Optional[list[str]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            copy: bool = False,
            *args,
            **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

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
    dimred_key = f"{uns_key}_diffmap"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    
    if f"X_{uns_key}_pca" not in adata.obsm or recalculate_pca:
        print("computing PCA for diffmap")
        sc.pp.pca(preprocessed_adata,
                  random_state = 187)
        adata = merge_pca_info_into_adata(adata,
                                          preprocessed_adata,
                                          dimred_key = f"{uns_key}_pca",
                                          uns_key = uns_key)

    if connectivities_key not in adata.obsp:
        print("computing neighbors for diffmap")
        sc.pp.neighbors(preprocessed_adata,
                        random_state = 187,
                        key_added = neighbors_key)
        try:
            adata = merge_neighbors_info_into_adata(adata,
                                                    preprocessed_adata,
                                                    neighbors_key = neighbors_key)
        except MemoryError:
            print("Neighbors information could not be merged due to a memory error")

 
    sc.tl.diffmap(preprocessed_adata,
                  neighbors_key = neighbors_key,
                  n_comps = 3,
                  *args,
                  **kwargs)

    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = preprocessed_adata,
                                                dimred = "diffmap",
                                                dimred_key = dimred_key)
    adata = add_uns_data(adata = adata,
                         gate_subset = preprocessed_adata,
                         old_key = "diffmap_evals",
                         key_added = f"{uns_key}_diffmap_evals")

    return adata if copy else None

def umap(adata: AnnData,
         gate: str,
         layer: Literal["compensated", "transformed"] = "transformed",
         recalculate_pca: bool = False,
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

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
    dimred_key = f"{uns_key}_umap"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"
    
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    if f"X_{uns_key}_pca" not in adata.obsm or recalculate_pca:
        print("computing PCA for umap")
        sc.pp.pca(preprocessed_adata,
                  random_state = 187)
        adata = merge_pca_info_into_adata(adata,
                                          preprocessed_adata,
                                          dimred_key = f"{uns_key}_pca",
                                          uns_key = uns_key)

    if connectivities_key not in adata.obsp:
        print("computing neighbors for umap")
        sc.pp.neighbors(preprocessed_adata,
                        random_state = 187,
                        key_added = neighbors_key)
        try:
            adata = merge_neighbors_info_into_adata(adata,
                                                    preprocessed_adata,
                                                    neighbors_key = neighbors_key)
        except MemoryError:
            print("Neighbors information could not be merged due to a memory error")
    
    sc.tl.umap(preprocessed_adata,
               neighbors_key = neighbors_key,
               n_components = 3,
               *args,
               **kwargs)
    
    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = preprocessed_adata,
                                                dimred = "umap",
                                                dimred_key = dimred_key)
    
    adata = add_uns_data(adata = adata,
                         gate_subset = preprocessed_adata,
                         old_key = "umap",
                         key_added = f"{uns_key}_umap")
  
    return adata if copy else None

def tsne(adata: AnnData,
         gate: str,
         layer: Literal["compensated", "transformed"] = "transformed",
         use_only_fluo: bool = True,
         exclude: Optional[list[str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         copy: bool = False,
         *args,
         **kwargs) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

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
    dimred_key = f"{uns_key}_tsne"

    sc.tl.tsne(preprocessed_adata,
               *args,
               **kwargs)

    adata = merge_dimred_coordinates_into_adata(adata = adata,
                                                gate_subset = preprocessed_adata,
                                                dimred = "tsne",
                                                dimred_key = dimred_key)
    
    adata = add_uns_data(adata = adata,
                         gate_subset = preprocessed_adata,
                         old_key = "tsne",
                         key_added = f"{uns_key}_tsne")

    return adata if copy else None
import parc as _parc
import scanpy as sc
from anndata import AnnData

from typing import Optional, Union, Literal

from ._utils import (preprocess_adata,
                    merge_cluster_info_into_adata,
                    merge_neighbors_info_into_adata)

def parc(adata: AnnData,
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
    cluster_key = f"{uns_key}_parc"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"

    preprocessed_adata = preprocess_adata(adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)

    ### we take the raw data as they are mostly below 50 markers anyway and would probably be not too much higher
    if connectivities_key not in adata.obsp:
        print("computing neighbors in parc")
        sc.pp.neighbors(preprocessed_adata,
                        random_state = 187,
                        key_added = neighbors_key)
        adata = merge_neighbors_info_into_adata(adata,
                                                preprocessed_adata,
                                                neighbors_key = neighbors_key)
 
    
    parcer = _parc.PARC(preprocessed_adata.layers[data_origin],
                        neighbor_graph = preprocessed_adata.obsp[connectivities_key]
                                         if connectivities_key in preprocessed_adata.obsp else None,
                        *args,
                        **kwargs)
    parcer.run_PARC()

    adata = merge_cluster_info_into_adata(adata,
                                          preprocessed_adata,
                                          cluster_key = cluster_key,
                                          cluster_assignments = parcer.labels)
       
    return adata if copy else None
import scanpy as sc
from anndata import AnnData

from typing import Optional, Union, Literal

from .utils import preprocess_adata

def leiden(adata: AnnData,
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
    cluster_key = f"{uns_key}_leiden"

    preprocessed_adata = preprocess_adata(adata,
                                          gate = gate,
                                          data_origin = data_origin,
                                          use_only_fluo = use_only_fluo,
                                          exclude = exclude,
                                          scaling = scaling)

    sc.tl.leiden(preprocessed_adata,
                 key_added = cluster_key,
                 *args,
                 **kwargs)

    adata.obs[cluster_key] = preprocessed_adata.obs[cluster_key].astype("category")

    return adata if copy else None
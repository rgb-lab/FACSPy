from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from .utils import reindex_dictionary, convert_to_dataframe

def calculate_positives(data: np.ndarray,
                        cofactors: np.ndarray) -> np.ndarray:
    positives = data > cofactors
    return positives.sum(axis = 0) / positives.shape[0]

def calculate_fop(adata: AnnData,
                  gates: Union[pd.Index, list[str]],
                  on: Literal["raw", "compensated", "transformed"],
                  cofactors: Optional[Union[np.ndarray, list[int], list[str]]]) -> dict[str, np.ndarray]:
    return {
        gate: calculate_positives(np.array(adata[adata.obsm["gating"][:,i] == 1,:].layers[on]),
                                    cofactors)
        for i, gate in enumerate(gates)
    }

def calculate_fops(adata: AnnData,
                   gates: Union[pd.Index, list[str]],
                   on: Literal["raw", "compensated", "transformed"],
                   groupby: Union[Literal["sample_ID"], str],
                   cofactors: Union[np.ndarray, list[int], list[str]]) -> dict:
    
    fops = {}
    for identifier in adata.obs[groupby].unique():
        tmp = adata[adata.obs[groupby] == identifier, :]
        fops[str(identifier)] = calculate_fop(adata = tmp,
                                              gates = gates,
                                              on = on,
                                              cofactors = cofactors)
    
    return fops

def fop(adata: AnnData,
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        cutoff: Optional[Union[int, float, list[int], list[float]]] = None,
        copy: bool = False):
    
    adata = adata.copy() if copy else adata
    gates = adata.uns["gating_cols"]
    cofactors = adata.var["cofactors"].to_numpy(dtype = np.float64) if cutoff is None else cutoff

    fops = calculate_fops(adata,
                          gates,
                          on = "compensated",
                          groupby = groupby,
                          cofactors = cofactors)
    adata.uns[f"fop_{groupby}_compensated"] = convert_to_dataframe(reindex_dictionary(fops), adata)

    if "transformed" in adata.layers:
        tfops = calculate_fops(adata,
                               gates,
                               on = "transformed",
                               groupby = groupby,
                               cofactors = cofactors)
        

        adata.uns[f"fop_{groupby}_transformed"] = convert_to_dataframe(reindex_dictionary(tfops), adata)


    return adata if copy else None
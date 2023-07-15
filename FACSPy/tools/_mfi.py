from typing import Union, Optional, Literal

from anndata import AnnData
import numpy as np
import pandas as pd

from .utils import reindex_dictionary, convert_to_dataframe

def calculate_medians(adata: AnnData,
                      gates: Union[pd.Index, list[str]],
                      on: Literal["raw", "compensated", "transformed"]) -> dict[str, np.ndarray]:
    return {
        gate: np.median(adata[adata.obsm["gating"][:,i] == 1,:].layers[on],
                        axis = 0) 
        for i, gate in enumerate(gates)
    }

def calculate_mfis(adata: AnnData,
                   gates: Union[pd.Index, list[str]],
                   on: Literal["raw", "compensated", "transformed"],
                   groupby: Union[Literal["sample_ID"], str]) -> dict:
    mfis = {}
    for identifier in adata.obs[groupby].unique():
        tmp = adata[adata.obs[groupby] == identifier, :]
        mfis[str(identifier)] = calculate_medians(adata = tmp,
                                                  gates = gates,
                                                  on = on)
    return mfis

def mfi(adata: AnnData,
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        copy: bool = False) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    gates = adata.uns["gating_cols"]

    mfis = calculate_mfis(adata,
                          gates,
                          on = "compensated",
                          groupby = groupby)
    adata.uns[f"mfi_{groupby}_compensated"] = convert_to_dataframe(reindex_dictionary(mfis), adata)

    if "transformed" in adata.layers:
        tmfis = calculate_mfis(adata,
                            gates,
                            on = "transformed",
                            groupby = groupby)
        

        adata.uns[f"mfi_{groupby}_transformed"] = convert_to_dataframe(reindex_dictionary(tmfis), adata)

    return adata if copy else None
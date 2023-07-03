from typing import Union, Optional, Literal
from anndata import AnnData
import numpy as np
import pandas as pd

def calculate_medians(adata: AnnData,
                      gates: Union[pd.Index, list[str]],
                      on: Literal["raw", "compensated", "transformed"]) -> dict[str, np.ndarray]:
    return {
        gate: np.median(adata[adata.obsm["gating"][:,i] == 1,:].layers[on],
                        axis = 0) 
        for i, gate in enumerate(gates)
    }

def reindex_dictionary(dictionary: dict) -> dict:
    ### reindexing the dictionary for multi-index in pandas    
    return {(outer_key, inner_key): values
            for outer_key, inner_dict in dictionary.items()
            for inner_key, values in inner_dict.items()}

def convert_to_dataframe(dictionary: dict,
                         adata: AnnData) -> pd.DataFrame:
    return pd.DataFrame(
            data = dictionary,
            index = adata.var.index,
            dtype = np.float32
        )

def calculate_mfis(adata: AnnData,
                   gates: Union[pd.Index, list[str]],
                   on = Literal["raw", "compensated", "transformed"]) -> dict:
    mfis = {}
    for sample_id in adata.obs["sample_ID"].unique():
        tmp = adata[adata.obs["sample_ID"] == sample_id, :]
        mfis[str(sample_id)] = calculate_medians(adata = tmp,
                                                 gates = gates,
                                                 on = on)
    return mfis

def mfi(adata: AnnData,
        population: Optional[Union[list[str], str]] = None,
        on: Literal["raw", "compensated", "transformed"] = "compensated",
        copy: bool = False):
    
    gates = adata.uns["gating_cols"]

    mfis = calculate_mfis(adata,
                          gates,
                          on = "compensated")
    adata.uns["mfi"] = convert_to_dataframe(reindex_dictionary(mfis), adata)

    tmfis = calculate_mfis(adata,
                           gates,
                           on = "transformed")
    adata.uns["tmfi"] = convert_to_dataframe(reindex_dictionary(tmfis), adata)

    return adata if copy else None
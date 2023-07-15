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

    # if groupby == "sample_ID":
    #     mfis = calculate_mfis(adata,
    #                           gates,
    #                           on = "compensated",
    #                           groupby = groupby)
        

    #     tmfis = calculate_mfis(adata,
    #                            gates,
    #                            on = "transformed",
    #                            groupby = groupby)
        

    #     adata.uns["tmfi"] = convert_to_dataframe(reindex_dictionary(tmfis), adata)
    #     adata.uns["mfi"] = convert_to_dataframe(reindex_dictionary(mfis), adata)
    
    # else:
    #     mfis_c = calculate_mfis(adata,
    #                             gates,
    #                             on = "compensated",
    #                             groupby = groupby)
    #     tmfis_c = calculate_mfis(adata,
    #                              gates,
    #                              on = "transformed",
    #                              groupby = groupby)
    #     adata.uns["tmfi_c"] = convert_to_dataframe(reindex_dictionary(tmfis_c), adata)
    #     adata.uns["mfi_c"] = convert_to_dataframe(reindex_dictionary(mfis_c), adata)  
    
    return adata if copy else None
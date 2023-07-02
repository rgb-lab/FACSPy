from typing import Union, Optional, Literal
import anndata as ad
import numpy as np
import pandas as pd

def mfi(dataset: ad.AnnData,
        population: Optional[Union[list[str], str]] = None,
        on: Literal["raw", "compensated", "transformed"] = "compensated",
        copy: bool = False):
    
    gates = dataset.uns["gating_cols"]

    mfis = {}
    for sample_id in dataset.obs["sample_ID"].unique():
        tmp = dataset[dataset.obs["sample_ID"] == sample_id, :]
        mfis[str(sample_id)] = {
                                    gate: np.median(tmp[tmp.obsm["gating"][:,i] == 1,:].layers[on],
                                                    axis = 0) 
                                    for i, gate in enumerate(gates)
                                }
    
    ### reindexing the dictionary for multi-index in pandas    
    mfis = {(outer_key, inner_key): values
            for outer_key, inner_dict in mfis.items()
            for inner_key, values in inner_dict.items()}
    mfi_frame = pd.DataFrame(
            data = mfis,
            index = dataset.var.index,
            dtype = np.float32
        )
    
    mfi_frame.source = f"{on} events"

    dataset.uns["mfi"] = mfi_frame

    return dataset if copy else None
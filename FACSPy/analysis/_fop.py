from typing import Union, Optional, Literal
import anndata as ad
import numpy as np
import pandas as pd

def calculate_positives(data: np.ndarray,
                        cofactors: np.ndarray) -> np.ndarray:
    positives = data > cofactors
    return positives.sum(axis = 0) / positives.shape[0]

def fop(dataset: ad.AnnData,
        population: Optional[Union[list[str], str]] = None,
        on: Literal["raw", "compensated", "transformed"] = "compensated",
        copy: bool = False):
    
    gates = dataset.uns["gating_cols"]
    cofactors = dataset.var["cofactors"].values

    fops = {}
    for sample_id in dataset.obs["sample_ID"].unique():
        tmp = dataset[dataset.obs["sample_ID"] == sample_id, :]
        fops[str(sample_id)] = {
                                    gate: calculate_positives(tmp[tmp.obsm["gating"][:,i] == 1,:].layers[on],
                                                              cofactors)
                                    for i, gate in enumerate(gates)
                                }
    
    ### reindexing the dictionary for multi-index in pandas    
    fops = {(outer_key, inner_key): values
            for outer_key, inner_dict in fops.items()
            for inner_key, values in inner_dict.items()}
    
    fop_frame = pd.DataFrame(
            data = fops,
            index = dataset.var.index,
            dtype = np.float16
        )
    
    fop_frame.source = f"{on} events"

    dataset.uns["fop"] = fop_frame

    return dataset if copy else None
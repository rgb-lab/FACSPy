from anndata import AnnData
import numpy as np
import pandas as pd

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
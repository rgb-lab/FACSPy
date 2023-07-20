from anndata import AnnData
import numpy as np
import pandas as pd
from typing import Literal

# def reindex_dictionary(dictionary: dict) -> dict:
#     ### reindexing the dictionary for multi-index in pandas    
#     return {(outer_key, inner_key): values
#             for outer_key, inner_dict in dictionary.items()
#             for inner_key, values in inner_dict.items()}

# def convert_to_dataframe(dictionary: dict,
#                          adata: AnnData) -> pd.DataFrame:
#     return pd.DataFrame(
#             data = dictionary,
#             index = adata.var.index,
#             dtype = np.float32
#         )

def assemble_dataframe(adata: AnnData,
                       on: Literal["transformed", "compensated"] = "compensated",
                       expression_data: bool = True) -> pd.DataFrame:
    obs = adata.obs.copy()
    gates = pd.DataFrame(data = adata.obsm["gating"].todense(),
                         columns = adata.uns["gating_cols"].to_list(),
                         index = obs.index)
    if expression_data:
        expression_data = adata.to_df(layer = on)
        return pd.concat([gates, expression_data, obs], axis = 1)
    return pd.concat([gates, obs], axis = 1)
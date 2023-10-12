from anndata import AnnData
import numpy as np
import pandas as pd
from typing import Literal, Optional

from ..utils import (contains_only_fluo,
                     subset_fluo_channels,
                     remove_channel,
                     subset_gate)

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

def scale_adata(adata: AnnData,
                scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]]) -> AnnData:
    if scaling == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        adata.X = MinMaxScaler().fit_transform(adata.X)
    elif scaling == "RobustScaler":
        from sklearn.preprocessing import RobustScaler
        adata.X = RobustScaler().fit_transform(adata.X)
    else:
        from sklearn.preprocessing import StandardScaler
        adata.X = StandardScaler().fit_transform(adata.X)
    return adata


def preprocess_adata(adata: AnnData,
                     gate: str,
                     data_origin: Literal["compensated", "transformed"],
                     use_only_fluo: bool = True,
                     exclude: Optional[list[str]] = None,
                     scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None) -> AnnData:
    adata = adata.copy()
    adata.X = adata.layers[data_origin]
    
    if scaling is not None:
        adata = scale_adata(adata,
                            scaling = scaling)
        
    if not contains_only_fluo(adata) and use_only_fluo:
        subset_fluo_channels(adata = adata)
    
    if exclude is not None:
        for channel in exclude:
            remove_channel(adata,
                           channel = channel,
                           copy = False)
    
    adata = subset_gate(adata = adata,
                        gate = gate,
                        as_view = True)
    assert adata.is_view
    return adata

from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ._utils import assemble_dataframe

# from .utils import reindex_dictionary, convert_to_dataframe

# def calculate_positives(data: np.ndarray,
#                         cofactors: np.ndarray) -> np.ndarray:
#     positives = data > cofactors
#     return positives.sum(axis = 0) / positives.shape[0]

# def calculate_fop(adata: AnnData,
#                   gates: Union[pd.Index, list[str]],
#                   on: Literal["raw", "compensated", "transformed"],
#                   cofactors: Optional[Union[np.ndarray, list[int], list[str]]]) -> dict[str, np.ndarray]:
#     return {
#         gate: calculate_positives(np.array(adata[adata.obsm["gating"][:,i] == 1,:].layers[on]),
#                                     cofactors)
#         for i, gate in enumerate(gates)
#     }

# def calculate_fops(adata: AnnData,
#                    gates: Union[pd.Index, list[str]],
#                    on: Literal["raw", "compensated", "transformed"],
#                    groupby: Union[Literal["sample_ID"], str],
#                    cofactors: Union[np.ndarray, list[int], list[str]]) -> dict:
    
#     fops = {}
#     for identifier in adata.obs[groupby].unique():
#         tmp = adata[adata.obs[groupby] == identifier, :]
#         fops[str(identifier)] = calculate_fop(adata = tmp,
#                                               gates = gates,
#                                               on = on,
#                                               cofactors = cofactors)
    
#     return fops

# def fop(adata: AnnData,
#         groupby: Union[Literal["sample_ID"], str] = "sample_ID",
#         cutoff: Optional[Union[int, float, list[int], list[float]]] = None,
#         copy: bool = False):
    
#     adata = adata.copy() if copy else adata
#     gates = adata.uns["gating_cols"]
#     cofactors = adata.var["cofactors"].to_numpy(dtype = np.float64) if cutoff is None else cutoff

#     fops = calculate_fops(adata,
#                           gates,
#                           on = "compensated",
#                           groupby = groupby,
#                           cofactors = cofactors)
#     adata.uns[f"fop_{groupby}_compensated"] = convert_to_dataframe(reindex_dictionary(fops), adata)

#     if "transformed" in adata.layers:
#         tfops = calculate_fops(adata,
#                                gates,
#                                on = "transformed",
#                                groupby = groupby,
#                                cofactors = cofactors)
        

#         adata.uns[f"fop_{groupby}_transformed"] = convert_to_dataframe(reindex_dictionary(tfops), adata)


#     return adata if copy else None


def calculate_fops_from_frame(input_frame: pd.DataFrame,
                              gate,
                              fluo_columns,
                              groupby: Optional[str],
                              cofactors: Union[pd.Series, int, float]) -> pd.DataFrame:
    groups = ["sample_ID", groupby] if groupby != "sample_ID" else ["sample_ID"]
    grouped_data = input_frame.loc[input_frame[gate] == True, fluo_columns + groups].groupby(groups, observed = True)
    data = grouped_data.sum() / grouped_data.count()
    data["gate"] = gate
    data = data.set_index(["gate"], append = True)
    return data.dropna(axis = 0, how = "all")

def fop(adata: AnnData,
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        cutoff: Optional[Union[int, float, list[int], list[float]]] = None,
        copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    fluo_columns = adata.var_names.to_list()
    cofactors = adata.var["cofactors"].to_numpy(dtype = np.float64) if cutoff is None else cutoff

    for data_subset in ["compensated", "transformed"]:
        dataframe = assemble_dataframe(adata,
                                       on = data_subset,
                                       expression_data = True)
        dataframe[fluo_columns] = dataframe[fluo_columns] > cofactors ## calculates positives as FI above cofactor
        adata.uns[f"fop_{groupby}_{data_subset}"] = pd.concat([calculate_fops_from_frame(dataframe, gate, fluo_columns, groupby, cofactors)
                                                               for gate in adata.uns["gating_cols"]])

    return adata if copy else None
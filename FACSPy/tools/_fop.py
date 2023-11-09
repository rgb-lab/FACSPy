from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ._utils import assemble_dataframe

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
        layer: list[str] = ["compensated", "transformed"],
        copy: bool = False) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata
    if not isinstance(layer, list):
        layer = [layer]
    fluo_columns = adata.var_names.to_list()
    cofactors = adata.var["cofactors"].to_numpy(dtype = np.float64) if cutoff is None else cutoff

    for _layer in layer:
        dataframe = assemble_dataframe(adata,
                                       on = _layer,
                                       expression_data = True)
        dataframe[fluo_columns] = dataframe[fluo_columns] > cofactors ## calculates positives as FI above cofactor
        adata.uns[f"fop_{groupby}_{_layer}"] = pd.concat([calculate_fops_from_frame(dataframe, gate, fluo_columns, groupby, cofactors)
                                                               for gate in adata.uns["gating_cols"]])

    return adata if copy else None
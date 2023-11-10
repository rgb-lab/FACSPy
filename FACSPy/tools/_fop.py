from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ._utils import _concat_gate_info_and_obs_and_fluo_data
from .._utils import fetch_fluo_channels
from ..dataset._utils import (_merge_cofactors_into_dataset_var,
                              _replace_missing_cofactors)

def _calculate_fops_from_frame(input_frame: pd.DataFrame,
                               gate,
                               fluo_columns,
                               groupby: Optional[str]) -> pd.DataFrame:
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
        use_only_fluo: bool = False,
        copy: bool = False) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    if not isinstance(layer, list):
        layer = [layer]
    
    if not "cofactors" in adata.var.columns:
        try:
            cofactor_table = adata.uns["cofactors"]
        except KeyError as e:
            raise e
        adata.var = _merge_cofactors_into_dataset_var(adata, cofactor_table)
        adata.var = _replace_missing_cofactors(adata.var)

    if use_only_fluo:
        columns_to_analyze = fetch_fluo_channels(adata)
    else:
        columns_to_analyze = adata.var_names.tolist()

    cofactors = adata.var.loc[columns_to_analyze, "cofactors"].to_numpy(dtype = np.float32) if cutoff is None else cutoff

    for _layer in layer:
        dataframe = _concat_gate_info_and_obs_and_fluo_data(adata,
                                                            on = _layer)

        dataframe[columns_to_analyze] = dataframe[columns_to_analyze] > cofactors ## calculates positives as FI above cofactor
        adata.uns[f"fop_{groupby}_{_layer}"] = pd.concat([_calculate_fops_from_frame(dataframe, gate, columns_to_analyze, groupby)
                                                               for gate in adata.uns["gating_cols"]])
        _save_settings(adata = adata,
                       groupby = groupby,
                       cutoff = cutoff,
                       cofactors = cofactors,
                       use_only_fluo = use_only_fluo,
                       layer = _layer)

    return adata if copy else None

def _save_settings(adata: AnnData,
                   groupby: str,
                   cutoff: Optional[Union[int, float, list[int], list[float]]],
                   cofactors: np.ndarray,
                   use_only_fluo: bool,
                   layer: str) -> None:

    if not "settings" in adata.uns:
        adata.uns["settings"] = {}
    
    adata.uns["settings"][f"_fop_{groupby}_{layer}"] = {
        "groupby": groupby,
        "cutoff": cutoff,
        "cofactors": cofactors,
        "use_only_fluo": use_only_fluo,
    }

    return 
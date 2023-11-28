from typing import Union, Optional, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ._utils import _concat_gate_info_and_obs_and_fluo_data
from .._utils import _fetch_fluo_channels, _default_layer
from ..dataset._utils import (_merge_cofactors_into_dataset_var,
                              _replace_missing_cofactors)

def _calculate_fops_from_frame(input_frame: pd.DataFrame,
                               gate,
                               fluo_columns,
                               groupby: Optional[str],
                               aggregate: bool) -> pd.DataFrame:
    if aggregate:
        groups = [groupby]
    else:
        groups = ["sample_ID", groupby] if groupby != "sample_ID" else [groupby]
    grouped_data = input_frame.loc[input_frame[gate] == True, fluo_columns + groups].groupby(groups, observed = True)
    data = grouped_data.sum() / grouped_data.count()
    data["gate"] = gate
    data = data.set_index(["gate"], append = True)
    return data.dropna(axis = 0, how = "all")

def _fop(adata: AnnData,
         layer: str,
         columns_to_analyze: list[str],
         cofactors: np.ndarray,
         groupby: Union[str, list[str]],
         aggregate: bool) -> pd.DataFrame:

    dataframe = _concat_gate_info_and_obs_and_fluo_data(adata,
                                                        layer = layer)
    dataframe[columns_to_analyze] = dataframe[columns_to_analyze] > cofactors ## calculates positives as FI above cofactor
    fop_frame = pd.concat([_calculate_fops_from_frame(dataframe,
                                                      gate,
                                                      columns_to_analyze,
                                                      groupby,
                                                      aggregate)
                            for gate in adata.uns["gating_cols"]])
    return fop_frame

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
        "cutoff": cutoff if cutoff is not None else cofactors,
        "use_only_fluo": use_only_fluo,
        "layer": layer
    }

    return 

@_default_layer
def fop(adata: AnnData,
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        cutoff: Optional[Union[int, float, list[int], list[float]]] = None,
        layer: Union[str, list[str]] = None,
        use_only_fluo: bool = False,
        aggregate: bool = False,
        copy: bool = False) -> Optional[AnnData]:

    adata = adata.copy() if copy else adata

    if not isinstance(layer, list):
        layer = [layer]
    
    if use_only_fluo:
        columns_to_analyze = _fetch_fluo_channels(adata)
    else:
        columns_to_analyze = adata.var_names.tolist()

    if cutoff is not None:
        cofactors = cutoff
    else:
        if not "cofactors" in adata.var.columns:
            try:
                cofactor_table = adata.uns["cofactors"]
            except KeyError as e:
                raise e
            adata.var = _merge_cofactors_into_dataset_var(adata, cofactor_table)
            adata.var = _replace_missing_cofactors(adata.var)

        cofactors = adata.var.loc[columns_to_analyze, "cofactors"].to_numpy(dtype = np.float32)

    for _layer in layer:

        fop_frame = _fop(adata = adata,
                         layer = _layer,
                         columns_to_analyze = columns_to_analyze,
                         cofactors = cofactors,
                         groupby = groupby,
                         aggregate = aggregate)
        
        adata.uns[f"fop_{groupby}_{_layer}"] = fop_frame

        _save_settings(adata = adata,
                       groupby = groupby,
                       cutoff = cutoff,
                       cofactors = cofactors,
                       use_only_fluo = use_only_fluo,
                       layer = _layer)

    return adata if copy else None
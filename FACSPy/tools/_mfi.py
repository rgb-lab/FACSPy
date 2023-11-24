from typing import Union, Optional, Literal

from anndata import AnnData
import pandas as pd

from ._utils import _concat_gate_info_and_obs_and_fluo_data
from .._utils import (_fetch_fluo_channels,
                      _default_layer,
                      _default_gate
                      )

def _mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.mean()

def _median(df: pd.DataFrame) -> pd.DataFrame:
    return df.median()

def _calculate_metric_from_frame(input_frame: pd.DataFrame,
                                 gate,
                                 fluo_columns,
                                 groupby: Optional[str],
                                 method: Literal["mean", "median"]) -> pd.DataFrame:
    groups = ["sample_ID", groupby] if groupby != "sample_ID" else ["sample_ID"]
    data = input_frame.loc[input_frame[gate] == True, fluo_columns + groups].groupby(groups, observed = True)
    if method == "mean":
        data = _mean(data)
    if method == "median":
        data = _median(data)
    data["gate"] = gate
    data = data.set_index(["gate"], append = True)
    return data.dropna(axis = 0, how = "all")

@_default_layer
def mfi(adata: AnnData,
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        layer: Union[str, list[str]] = None,
        method: Literal["mean", "median"] = "median",
        use_only_fluo: bool = False,
        copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata

    if not isinstance(layer, list):
        layer = [layer]
    
    if method not in ["median", "mean"]:
        raise NotImplementedError("metric must be one of ['median', 'mean', 'geo_mean']")

    if use_only_fluo:
        columns_to_analyze = _fetch_fluo_channels(adata)
    else:
        columns_to_analyze = adata.var_names.tolist()

    for _layer in layer:
        dataframe = _concat_gate_info_and_obs_and_fluo_data(adata,
                                                            layer = _layer)
        
        adata.uns[f"mfi_{'_'.join([groupby])}_{_layer}"] = pd.concat([_calculate_metric_from_frame(dataframe,
                                                                                                   gate,
                                                                                                   columns_to_analyze,
                                                                                                   groupby,
                                                                                                   method)
                                                                      for gate in adata.uns["gating_cols"]])
        _save_settings(adata = adata,
                       groupby = groupby,
                       method = method,
                       use_only_fluo = use_only_fluo,
                       layer = _layer)

    return adata if copy else None

def _save_settings(adata: AnnData,
                   groupby: str,
                   method: Literal["mean", "median"],
                   use_only_fluo: bool,
                   layer: str) -> None:

    if not "settings" in adata.uns:
        adata.uns["settings"] = {}
    
    adata.uns["settings"][f"_mfi_{groupby}_{layer}"] = {
        "groupby": groupby,
        "method": method,
        "use_only_fluo": use_only_fluo,
        "layer": layer
    }

    return 
from typing import Union, Optional, Literal

from anndata import AnnData
import pandas as pd

from .utils import assemble_dataframe

def calculate_medians_from_frame(input_frame: pd.DataFrame,
                                 gate,
                                 fluo_columns,
                                 groupby: Optional[str]) -> pd.DataFrame:
    #groups = ["sample_ID", groupby] if groupby != "sample_ID" else ["sample_ID"]
    data = input_frame.loc[input_frame[gate] == True, fluo_columns + groupby].groupby(groupby).median()
    data["gate"] = gate
    data = data.set_index(["gate"], append = True)
    return data.dropna(axis = 0, how = "all")

def mfi(adata: AnnData,
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata

    if not isinstance(groupby, list):
        groupby = [groupby]

    fluo_columns = adata.var_names.to_list()

    for data_subset in ["compensated", "transformed"]:
        dataframe = assemble_dataframe(adata,
                                       on = data_subset,
                                       expression_data = True)
        
        adata.uns[f"mfi_{'_'.join(groupby)}_{data_subset}"] = pd.concat([calculate_medians_from_frame(dataframe, gate, fluo_columns, groupby)
                                                               for gate in adata.uns["gating_cols"]])

    return adata if copy else None


import numpy as np
import pandas as pd

from anndata import AnnData

from typing import Union, Optional, Literal
from ..utils import find_gate_path_of_gate, create_comparisons

from ..exceptions.exceptions import AnalysisNotPerformedError

from ..plotting.utils import (prep_uns_dataframe,
                              select_gate_from_singleindex_dataframe)

from scipy.stats import kruskal

def calculate_asinh_fold_change(data: pd.DataFrame,
                                groupby: str,
                                group1: str,
                                group2: str,
                                fluo_columns: list[str]) -> pd.DataFrame:
    grouped = data.groupby(groupby).mean(fluo_columns)
    grouped = grouped.loc[[group1, group2], fluo_columns]
    grouped.loc["asinh_fc", :] = np.arcsinh([grouped.loc[group2,:]]) - np.arcsinh([grouped.loc[group1,:]])
    return grouped.T

def calculate_p_values(data: pd.DataFrame,
                       groupby: str,
                       group1: str,
                       group2: str,
                       fluo_columns: list[str],
                       n_comparisons: int,
                       test: str) -> pd.DataFrame:
    p_frame = pd.DataFrame(index = fluo_columns,
                           columns = ["p", "p_adj"])
    for marker in fluo_columns:
        group_1 = data.loc[data[groupby] == group1, marker]
        group_2 = data.loc[data[groupby] == group2, marker]
        p_value = kruskal(group_1, group_2).pvalue
        p_frame.loc[marker, "p"] = p_value
        p_frame.loc[marker, "p_adj"] = p_value * n_comparisons

    return p_frame



def calculate_fold_changes(adata: AnnData,
                           groupby: str,
                           group1: str,
                           group2: str,
                           gate: str,
                           on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                           test: Literal["Kruskal", "t-test"] = "Kruskal"
                           ) -> pd.DataFrame:
    
    """asinh fold change calculation"""

    try:
        data = adata.uns[on]
        data = prep_uns_dataframe(adata, data)
        data = select_gate_from_singleindex_dataframe(data, find_gate_path_of_gate(adata, gate))
        fluo_columns = [col for col in data.columns if col in adata.var_names.to_list()]

    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e
    
    #comparisons = create_comparisons(data, groupby)

    print(group1, group2)

    asinh_fc = calculate_asinh_fold_change(data,
                                            groupby,
                                            group1,
                                            group2,
                                            fluo_columns)
    p_values = calculate_p_values(data,
                                  groupby,
                                  group1,
                                  group2,
                                  fluo_columns,
                                  n_comparisons = len(data[groupby].unique()),
                                  test = test)
        
        
    return pd.merge(asinh_fc, p_values, left_index = True, right_index = True)
    





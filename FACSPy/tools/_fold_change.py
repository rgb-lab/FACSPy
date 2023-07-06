import numpy as np
import pandas as pd

from anndata import AnnData

from typing import Union, Optional, Literal
from ..utils import find_gate_path_of_gate, create_comparisons

from ..exceptions.exceptions import AnalysisNotPerformedError, NotSupportedStatisticalTestError

from ..plotting.utils import (get_uns_dataframe,
                              select_gate_from_singleindex_dataframe)

from scipy.stats import kruskal, wilcoxon

def calculate_asinh_fold_change(data: pd.DataFrame,
                                groupby: str,
                                group1: list[Union[str, int]],
                                group2: list[Union[str, int]],
                                fluo_columns: list[str]) -> pd.DataFrame:
    """ 
    calculates the asinh fold change by getting subtracting the arcsinh MFIs
    If multiple groups have been defined, the mean of both is used for calculation
    """
    grouped = data.groupby(groupby).mean(fluo_columns)
    grouped = grouped.loc[group1 + group2, fluo_columns]
    grouped.loc["group1",:] = np.mean(grouped.loc[group1,:], axis = 0)
    grouped.loc["group2",:] = np.mean(grouped.loc[group2,:], axis = 0)
    grouped = grouped.drop(group1 + group2, axis = 0)
    grouped.loc["asinh_fc", :] = np.arcsinh([grouped.loc["group2",:]]) - np.arcsinh([grouped.loc["group1",:]])
    return grouped.T

def calculate_p_values(data: pd.DataFrame,
                       groupby: str,
                       group1: list[Union[str, int]],
                       group2: list[Union[str, int]],
                       fluo_columns: list[str],
                       n_comparisons: int,
                       test: str) -> pd.DataFrame:
    """ 
    calculates the asinh fold change by getting subtracting the arcsinh MFIs
    If multiple groups have been defined, both values are used for the p_value calculation
    """  
    p_frame = pd.DataFrame(index = fluo_columns,
                           columns = ["p", "p_adj"])
    
    for marker in fluo_columns:
        group_1 = data.loc[data[groupby].isin(group1), marker].to_numpy()
        group_2 = data.loc[data[groupby].isin(group2), marker].to_numpy()
        p_value = calculate_pvalue(group_1,
                                   group_2,
                                   test)
        p_frame.loc[marker, "p"] = p_value
        p_frame.loc[marker, "p_adj"] = p_value * n_comparisons

    return p_frame

def calculate_pvalue(group1: np.ndarray,
                     group2: np.ndarray,
                     test: str) -> float:
    if test == "Kruskal":
        return kruskal(group1, group2).pvalue
    if test == "Wilcoxon":
        return wilcoxon(group1, group2).pvalue
    available_tests = ["Kruskal", "Wilcoxon"]
    raise NotSupportedStatisticalTestError(test, available_tests)

def calculate_fold_changes(adata: AnnData,
                           groupby: str,
                           group1: Union[str, list[Union[str, int]]],
                           group2: Union[str, list[Union[str, int]]],
                           gate: str,
                           on: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                           test: Literal["Kruskal", "t-test"] = "Kruskal"
                           ) -> pd.DataFrame:
    
    """asinh fold change calculation"""
    if not isinstance(group1, list):
        group1 = [group1]
    
    if not isinstance(group2, list):
        group2 = [group2]
    
    data = get_uns_dataframe(adata = adata,
                             gate = gate,
                             table_identifier = "mfi",
                             column_identifier_name = "sample_ID")
    fluo_columns = [col for col in data.columns if col in adata.var_names]
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
    





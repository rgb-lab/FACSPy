import warnings
import numpy as np
import pandas as pd

from anndata import AnnData

from typing import Union, Literal, Optional

from ..exceptions._exceptions import NotSupportedStatisticalTestError, NaNRemovalWarning

from ..plotting._utils import _get_uns_dataframe

from scipy.stats import kruskal, wilcoxon

def calculate_asinh_fold_change(data: pd.DataFrame,
                                groupby: str,
                                group1: list[Union[str, int]],
                                group2: list[Union[str, int]],
                                fluo_columns: list[str],
                                cofactors: list[float]) -> pd.DataFrame:
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
    if np.any(np.isnan(group1)):
        warnings.warn("NaN detected while calculating fold changes. The NaN will be removed!",
                      NaNRemovalWarning)
        group1 = group1[~np.isnan(group1)]

    if np.any(np.isnan(group2)):
        warnings.warn("NaN detected while calculating fold changes. The NaN will be removed!",
                      NaNRemovalWarning)
        group2 = group2[~np.isnan(group2)]

    if test == "Kruskal":
        try:
            return kruskal(group1, group2).pvalue
        except ValueError:
            print("Warning! Had a kruskal error message, potentially for all values are the same!")
            return 1
    if test == "Wilcoxon":
        return wilcoxon(group1, group2).pvalue
    available_tests = ["Kruskal", "Wilcoxon"]
    raise NotSupportedStatisticalTestError(test, available_tests)

def _calculate_fold_changes(adata: AnnData,
                            groupby: str,
                            group1: Union[list[Union[str, int]], str, int],
                            group2: Union[list[Union[str, int]], str, int],
                            gate: str,
                            layer: str,
                            data_group: Optional[Union[list[str], str]] = "sample_ID",
                            data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                            test: Literal["Kruskal", "t-test"] = "Kruskal"
                            ) -> pd.DataFrame:
    
    """asinh fold change calculation"""
    if not isinstance(group1, list):
        group1 = [group1]
    
    if not isinstance(group2, list):
        group2 = [group2]
    
    data = _get_uns_dataframe(adata = adata,
                              gate = gate,
                              table_identifier = f"{data_metric}_{data_group}_{layer}")
    
    channel_columns = [col for col in data.columns if col in adata.var_names]
    cofactors = adata.var.loc[channel_columns, "cofactors"].astype("float32")
    data[channel_columns] = data[channel_columns].divide(cofactors)

    asinh_fc = calculate_asinh_fold_change(data,
                                           groupby,
                                           group1,
                                           group2,
                                           channel_columns,
                                           cofactors)
    p_values = calculate_p_values(data,
                                  groupby,
                                  group1,
                                  group2,
                                  channel_columns,
                                  n_comparisons = len(data[groupby].unique()),
                                  test = test)
        
        
    return pd.merge(asinh_fc,
                    p_values,
                    left_index = True,
                    right_index = True)

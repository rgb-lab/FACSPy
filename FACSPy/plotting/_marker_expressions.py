import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from typing import Literal, Union, Optional

from .cofactor_plots import calculate_histogram_data

def flatten_nested_list(l):
    return [item for sublist in l for item in sublist]

def convert_to_mapping(dataframe: pd.DataFrame,
                       key_col: str,
                       value_col: str) -> dict:
    return {key_value: dataframe.loc[dataframe[key_col] == key_value, value_col].iloc[0] for key_value in dataframe[key_col].unique()}

def append_metadata_obs(adata: AnnData,
                        expression_data: pd.DataFrame) -> pd.DataFrame:
    expression_data[adata.obs.columns] = adata.obs
    return expression_data

def convert_expression_to_histogram_data(expression_data: pd.DataFrame,
                                         marker: str,
                                         groupby: str) -> pd.DataFrame:
    group_values = expression_data[groupby].unique()
    
    histogram_df = pd.DataFrame(
        data = {groupby: flatten_nested_list([[group for _ in range (100)] for group in group_values])},
        columns = [groupby, "x", "y"],
        index = range(100 * len(group_values))
    )
    
    for group in  group_values:
        group_spec_expression_data = expression_data.loc[expression_data[groupby] == group, [groupby, marker]]
        x, y = calculate_histogram_data(group_spec_expression_data,
                                        {"x": marker})
        histogram_df.loc[histogram_df[groupby] == group, ["x", "y"]] = np.vstack([x, y]).T

    return histogram_df

def append_colorby_variable(adata: AnnData,
                            dataframe: pd.DataFrame,
                            colorby: str) -> pd.DataFrame:
    mapping = convert_to_mapping(adata.uns["metadata"].to_df(), key_col = "sample_ID", value_col = colorby)
    dataframe[colorby] = dataframe["sample_ID"].map(mapping)
    return dataframe

def marker_expression_samplewise(adata: AnnData,
                                 markers: Union[str, list[str]],
                                 colorby: Union[str, list[str]],
                                 on: Literal["compensated", "transformed", "raw"] = "transformed") -> Optional[Figure]:
    """plots histograms per sample and colors by colorby variable"""
    if not isinstance(marker, list):
        marker = [marker]

    if not isinstance(groupby, list):
        groupby = [groupby]

    expression_data = adata.to_df(layer = on)
    expression_data = append_metadata_obs(adata, expression_data)

    ## pot. buggy: groupby only singular at the moment...
    for marker in markers:
        histogram_df = convert_expression_to_histogram_data(expression_data = expression_data,
                                                            marker = marker,
                                                            groupby = "sample_ID")
        histogram_df = append_colorby_variable(adata = adata,
                                               dataframe = histogram_df,
                                               colorby = colorby)
        



    
from anndata import AnnData
import pandas as pd
from typing import Union, Literal

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib import pyplot as plt
import seaborn as sns

from ..utils import find_gate_path_of_gate

from ..exceptions.exceptions import AnalysisNotPerformedError

from .utils import create_boxplot
### mfi plot

def append_metadata(adata: AnnData,
                    mfi_data: pd.DataFrame) -> pd.DataFrame:
    mfi_data = mfi_data.T
    mfi_data.index = mfi_data.index.set_names(["sample_ID", "gate_path"])
    mfi_data["sample_ID"] = mfi_data["sample_ID"].astype("int64")
    metadata = adata.uns["metadata"].to_df()

    return pd.merge(mfi_data, metadata, on = "sample_ID")




def mfi(adata: AnnData,
        markers: Union[str, list[str]],
        groupby = Union[str, list[str]],
        gate: str = None):

    try:
        mfi_data = adata.uns["mfi"]
    except KeyError:
        raise AnalysisNotPerformedError("mfi")

    if gate is None:
        raise TypeError("A Gate has to be provided")
    
    if not isinstance(markers):
        markers = [markers]
    if not isinstance(groupby, list):
        groupby = [groupby]

    ncols = 1,
    nrows = len(groupby)
    figsize = (3, 3 * len(groupby)) 
    
    fig, ax = plt.subplots(ncols = 1, nrows = nrows, figsize = figsize)


    full_gate_path = find_gate_path_of_gate(adata, gate)
    gate_specific_mfis = mfi_data.loc[mfi_data["gate_path"] == full_gate_path, :]
    for grouping in groupby:
        for marker in markers:
            plot_params = {
                "x": "sample_ID" if grouping is None else grouping,
                "y": marker,
                    "data": gate_specific_mfis
            }

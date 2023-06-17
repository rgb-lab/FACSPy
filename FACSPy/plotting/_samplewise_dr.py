import pandas as pd
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
import seaborn as sns

from matplotlib.axis import Axis
from matplotlib.figure import Figure

from typing import Literal, Union, Optional

from .utils import prep_uns_dataframe

from ..exceptions.exceptions import AnalysisNotPerformedError


def samplewise_dr_plot(adata: AnnData,
                       dataframe: pd.DataFrame,
                       markers: Optional[Union[str, list[str]]],
                       groupby: Optional[Union[str, list[str]]],
                       reduction: Literal["PCA", "MDS", "TSNE", "UMAP"],
                       gate: Optional[str] = None,
                       overview: bool = False):
    
    if gate is None:
        raise TypeError("A Gate has to be provided")
    if overview:
        if markers:
            print("warning... markers and groupby argument are ignored when using overview")
        markers = adata.var_names.to_list()
        groupby = adata.obs.columns.to_list()
    
    if markers is None:
        markers = []
    if groupby is None:
        groupby = []   
    
    if not isinstance(markers, list):
        markers = [markers]
    if not isinstance(groupby, list):
        groupby = [groupby]

    full_gate_path = find_gate_path_of_gate(adata, gate)
    gate_specific_mfis = dataframe.loc[dataframe["gate_path"] == full_gate_path, :]

    for i, grouping in enumerate(groupby + markers):
        

        

    

def pca_samplewise(adata: AnnData,
                   groupby: str,
                   markers: str,
                   gate: str, 
                   on: Literal["mfi", "fop", "gate_frequency"] = "mfi"
                   ) -> Optional[Figure]:
    
    try:
        data = adata.uns[on]
        data = prep_uns_dataframe(adata, data)
    except KeyError as e:
        raise AnalysisNotPerformedError(on) from e
    
    samplewise_dr_plot(adata = adata,
                       groupby = groupby,
                       dataframe = data,
                       reduction = "PCA",
                       markers = markers,
                       gate = gate,
                       overview = overview)
    
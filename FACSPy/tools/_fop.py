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
        layer: Union[str, list[str]] = None,
        cutoff: Optional[Union[int, float, list[int], list[float]]] = None,
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        use_only_fluo: bool = False,
        aggregate: bool = False,
        copy: bool = False) -> Optional[AnnData]:
    """\
    
    Calculates the frequency of parents. For the calculation, cutoffs
    per channel are necessary that can be supplied as a singular value,
    a list of values or via the cofactors. FOPs are calculated for all
    gates.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input. Multiple layers can be passed
        as a list.
    cutoff
        Intensity cutoff above which cells are counted as positive. Can be
        a singular value, a value per channel. If None, values will be pulled
        from `.var['cofactors']` or from `.uns['cofactors']`.
    groupby
        Argument to specify the grouping. Defaults to sample_ID, which
        will calculate the FOP per sample. Can be any column from the `.obs`
        slot of adata.
    use_only_fluo
        Parameter to specify if the FOP should only be calculated for the fluorescence
        channels. Defaults to False.
    aggregate
        If False, calculates the FOP as if `groupby==['sample_ID', groupby]`. If set to
        True, the FOP is calculated per entry in `.obs[groupby]`. Defaults to False.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.uns[f'fop_{groupby}_{layer}']`
            calculated FOP values per channel
        `.uns['settings'][f'_fop_{groupby}_{layer}']`
            Settings that were used for calculation

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn', 'cofactors'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.tl.fop(dataset)

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition', 'sex'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn', 'cofactors'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.settings.default_gate = "T_cells"
    >>> fp.settings.default_layer = "transformed"
    >>> fp.tl.pca(dataset)
    >>> fp.tl.neighbors(dataset)
    >>> fp.tl.leiden(dataset)
    >>> fp.tl.fop(dataset, groupby = "leiden", aggregate = True) # will calculate FOP per leiden cluster
    >>> fp.tl.fop(dataset, groupby = "leiden", aggregate = False) # will calculate FOP per leiden cluster and sample_ID
    
    """

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
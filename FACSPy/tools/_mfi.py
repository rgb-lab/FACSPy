from typing import Union, Optional, Literal

from anndata import AnnData
import pandas as pd

from ._utils import _concat_gate_info_and_obs_and_fluo_data
from .._utils import (_fetch_fluo_channels,
                      _default_layer)

def _mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.mean()

def _median(df: pd.DataFrame) -> pd.DataFrame:
    return df.median()

def _calculate_metric_from_frame(input_frame: pd.DataFrame,
                                 gate: str,
                                 fluo_columns: Union[list[str], str],
                                 groupby: Union[list[str], str],
                                 method: Literal["mean", "median"],
                                 aggregate: bool) -> pd.DataFrame:
    if aggregate:
        groups = [groupby]
    else:
        groups = ["sample_ID", groupby] if groupby != "sample_ID" else [groupby]
    if not isinstance(fluo_columns, list):
        fluo_columns = [fluo_columns]
    data = input_frame.loc[input_frame[gate] == True,
                           fluo_columns + groups].groupby(groups, observed = True)
    if method == "mean":
        data = _mean(data)
    if method == "median":
        data = _median(data)
    data["gate"] = gate
    data = data.set_index(["gate"], append = True)
    return data.dropna(axis = 0, how = "all")

def _save_settings(adata: AnnData,
                   groupby: str,
                   method: Literal["mean", "median"],
                   use_only_fluo: bool,
                   layer: str) -> None:

    if "settings" not in adata.uns:
        adata.uns["settings"] = {}
    
    adata.uns["settings"][f"_mfi_{groupby}_{layer}"] = {
        "groupby": groupby,
        "method": method,
        "use_only_fluo": use_only_fluo,
        "layer": layer
    }

    return 

def _mfi(adata: AnnData,
         layer: str,
         columns_to_analyze: list[str],
         groupby: Union[list[str], str],
         method: Literal["mean", "median"],
         aggregate: bool) -> pd.DataFrame:

    dataframe = _concat_gate_info_and_obs_and_fluo_data(adata,
                                                        layer = layer)
    mfi_frame = pd.concat([_calculate_metric_from_frame(dataframe,
                                                        gate,
                                                        columns_to_analyze,
                                                        groupby,
                                                        method,
                                                        aggregate)
                            for gate in adata.uns["gating_cols"]])
    return mfi_frame

@_default_layer
def mfi(adata: AnnData,
        layer: Union[list[str], str],
        method: Literal["mean", "median"] = "median",
        groupby: Union[Literal["sample_ID"], str] = "sample_ID",
        use_only_fluo: bool = False,
        aggregate: bool = False,
        copy: bool = False) -> Optional[AnnData]:
    """\
    
    Calculates the median/mean fluorescence intensities (MFI).
    MFIs are calculated for all gates at once.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input. Multiple layers can be passed
        as a list.
    method
        Whether to use `mean` or `median` for the calculation. Defaults to median.
    groupby
        Argument to specify the grouping. Defaults to sample_ID, which
        will calculate the FOP per sample. Can be any column from the `.obs`
        slot of adata.
    use_only_fluo
        Parameter to specify if the MFI should only be calculated for the fluorescence
        channels.
    aggregate
        If False, calculates the MFI as if `groupby==['sample_ID', groupby]`. If set to
        True, the FOP is calculated per entry in `.obs[groupby]`. Defaults to False.
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.uns[f'mfi_{groupby}_{layer}']`
            calculated MFI values per channel
        `.uns['settings'][f'_mfi_{groupby}_{layer}']`
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
    >>> fp.tl.mfi(dataset)

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
    >>> fp.tl.mfi(dataset,
    ...           groupby = "T_cells_transformed_leiden",
    ...           aggregate = True) # will calculate MFI per leiden cluster
    >>> fp.tl.mfi(dataset,
    ...           groupby = "T_cells_transformed_leiden",
    ...           aggregate = False) # will calculate MFI per leiden cluster and sample_ID
    
    """

    adata = adata.copy() if copy else adata

    if not isinstance(layer, list):
        layer = [layer]
    
    if method not in ["median", "mean"]:
        raise NotImplementedError("metric must be one of ['median', 'mean']")

    if use_only_fluo:
        columns_to_analyze = _fetch_fluo_channels(adata)
    else:
        columns_to_analyze = adata.var_names.tolist()

    for _layer in layer:
        mfi_frame = _mfi(adata = adata,
                         layer = _layer,
                         columns_to_analyze = columns_to_analyze,
                         groupby = groupby,
                         method = method,
                         aggregate = aggregate)

        adata.uns[f"mfi_{'_'.join([groupby])}_{_layer}"] = mfi_frame

        _save_settings(adata = adata,
                       groupby = groupby,
                       method = method,
                       use_only_fluo = use_only_fluo,
                       layer = _layer)

    return adata if copy else None

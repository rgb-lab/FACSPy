from anndata import AnnData

from typing import Literal, Union, Optional

from ._dr_samplewise import _perform_samplewise_dr
from .._utils import _default_layer

@_default_layer
def mds_samplewise(adata: AnnData,
                   layer: str,
                   data_group: str = "sample_ID",
                   data_metric: Literal["mfi", "fop"] = "mfi",
                   use_only_fluo: bool = True,
                   exclude: Optional[Union[list[str], str]] = None,
                   scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",
                   n_components: int = 3,
                   copy: bool = False,
                   *args,
                   **kwargs) -> Optional[AnnData]:
    """\
    Computes samplewise MDS based on either the median fluorescence values (MFI)
    or frequency of parent values (FOP). MDS will be calculated for all gates at once.
    The values are added to the corresponding `.uns` slot where MFI/FOP values are
    stored.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    n_components
        The number of components to be calculated. Defaults to 3.
    use_only_fluo
        Parameter to specify if the MDS should only be calculated for the fluorescence
        channels.
    exclude
        Can be used to exclude channels from calculating the embedding.
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score). Defaults to None.
    data_metric
        One of `mfi` or `fop`. Using a different metric will calculate
        the asinh fold change on mfi and fop values, respectively
    data_group
        When MFIs/FOPs are calculated, and the groupby parameter is used,
        use `data_group` to specify the right dataframe
    copy
        Return a copy of adata instead of modifying inplace.
    **kwargs : dict, optional
        keyword arguments that are passed directly to the `sklearn.MDS`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.uns[f'{data_metric}_{data_group}_{layer}']`
            MDS coordinates are added to the respective frame
        `.uns['settings'][f"_mds_samplewise_{data_metric}_{layer}"]`
            Settings that were used for samplewise MDS calculation

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
    >>> fp.settings.default_gate = "T_cells"
    >>> fp.settings.default_layer = "transformed"
    >>> fp.tl.mfi(dataset)
    >>> fp.tl.mds_samplewise(dataset)

    """

    adata = adata.copy() if copy else adata

    adata = _perform_samplewise_dr(adata = adata,
                                   reduction = "MDS",
                                   data_metric = data_metric,
                                   data_group = data_group,
                                   layer = layer,
                                   use_only_fluo = use_only_fluo,
                                   exclude = exclude,
                                   scaling = scaling,
                                   n_components = n_components,
                                   *args,
                                   **kwargs)

    return adata if copy else None


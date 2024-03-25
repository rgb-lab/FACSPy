
from anndata import AnnData
from FlowSOM import flowsom as _flowsom

from typing import Optional, Literal, Union

from ._utils import (_preprocess_adata,
                     _merge_cluster_info_into_adata,
                     _save_cluster_settings)

from .._utils import (_default_gate_and_default_layer,
                      _enable_gate_aliases,
                      IMPLEMENTED_SCALERS)
from ..exceptions._exceptions import InvalidScalingError

@_default_gate_and_default_layer
@_enable_gate_aliases
def flowsom(adata: AnnData,
            gate: str,
            layer: str,
            key_added: Optional[str] = None,
            use_only_fluo: bool = True,
            exclude: Optional[Union[list[str], str]] = None,
            scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
            copy: bool = False,
            **kwargs) -> Optional[AnnData]:
    """\
    Computes FlowSOM clustering. 

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    key_added
        Name of the `.obs` column that is filled with the cluster
        annotations. Defaults to f'{gate}_{layer}_flowsom'.
    use_only_fluo
        Parameter to specify if the UMAP should only be calculated for the fluorescence
        channels. Specify `recalculate_pca` to repeat PCA calculation.
    exclude
        Can be used to exclude channels from calculating the embedding.
        Specify `recalculate_pca` to repeat PCA calculation.
    scaling
        Whether to apply scaling to the data for display. One of `MinMaxScaler`,
        `RobustScaler` or `StandardScaler` (Z-score). Defaults to None.
    copy
        Return a copy of adata instead of modifying inplace
    **kwargs : dict, optional
        keyword arguments that are passed directly to the `flowsom.flowsom`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.obs[f'{gate}_{layer}_flowsom]`
            cluster annotations
        `.uns['settings'][f'_flowsom_{gate}_{layer}]`
            Settings that were used for FlowSOM calculation

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
    >>> fp.tl.pca(dataset)
    >>> fp.tl.neighbors(dataset)
    >>> fp.tl.flowsom(dataset)

    """
    
    adata = adata.copy() if copy else adata

    if exclude is None:
        exclude = []
    else:
        if not isinstance(exclude, list):
            exclude = [exclude]

    if scaling not in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    uns_key = f"{gate}_{layer}"
    cluster_key = key_added or f"{uns_key}_flowsom"

    if not kwargs:
        from multiprocessing import cpu_count
        kwargs = {
            "x_dim": 50,
            "y_dim": 50,
            "n_jobs": max(1, cpu_count() - 2)
        }

    if "consensus_cluster_max_n" not in kwargs:
        kwargs["consensus_cluster_max_n"] = min(50, adata.shape[0])

    _save_cluster_settings(adata = adata,
                           gate = gate,
                           layer = layer,
                           use_only_fluo = use_only_fluo,
                           exclude = exclude,
                           scaling = scaling,
                           clustering = "flowsom",
                           **kwargs)

    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    cluster_annotations = _flowsom(preprocessed_adata.X,
                                   **kwargs)

    adata = _merge_cluster_info_into_adata(adata,
                                           preprocessed_adata,
                                           cluster_key = cluster_key,
                                           cluster_assignments = cluster_annotations)
    
    return adata if copy else None

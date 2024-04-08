import phenograph as _phenograph
from anndata import AnnData

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
def phenograph(adata: AnnData,
               gate: str,
               layer: str,
               key_added: Optional[str] = None,
               use_only_fluo: bool = True,
               exclude: Optional[Union[list[str], str]] = None,
               scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
               copy: bool = False,
               **kwargs) -> Optional[AnnData]:
    """\
    Computes PhenoGraph clustering. 

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
        annotations. Defaults to f'{gate}_{layer}_phenograph'.
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
        keyword arguments that are passed directly to the `phenograph.cluster`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.obs[f'{gate}_{layer}_phenograph]`
            cluster annotations
        `.uns['settings'][f'_phenograph_{gate}_{layer}]`
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
    >>> fp.tl.phenograph(dataset)

    """
    
    adata = adata.copy() if copy else adata

    if exclude is None:
        exclude = []
    else:
        if not isinstance(exclude, list):
            exclude = [exclude]

    if scaling not in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    if "clustering_algo" not in kwargs:
        kwargs["clustering_algo"] = "leiden"

    _save_cluster_settings(adata = adata,
                           gate = gate,
                           layer = layer,
                           use_only_fluo = use_only_fluo,
                           exclude = exclude,
                           scaling = scaling,
                           clustering = "phenograph",
                           **kwargs)

    uns_key = f"{gate}_{layer}"
    cluster_key = key_added or f"{uns_key}_phenograph"

    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)
    if ("k" in kwargs and kwargs["k"] <= preprocessed_adata.shape[0]) or "k" not in kwargs:
        print(f"warning! Setting k to {min(preprocessed_adata.shape[0] - 1, 30)} to avoid errors")
        kwargs["k"] = min(preprocessed_adata.shape[0]-2, 30)

    communities, graph, Q = _phenograph.cluster(preprocessed_adata.X,
                                                **kwargs)

    adata = _merge_cluster_info_into_adata(adata,
                                           preprocessed_adata,
                                           cluster_key = cluster_key,
                                           cluster_assignments = communities)

    adata.uns[f"{cluster_key}_graph"] = graph
    adata.uns[f"{cluster_key}_Q"] = Q

    return adata if copy else None

import parc as _parc
from anndata import AnnData

from typing import Optional, Union, Literal

from ._utils import (_preprocess_adata,
                     _merge_cluster_info_into_adata,
                     _extract_valid_neighbors_kwargs,
                     _extract_valid_parc_kwargs,
                     _save_cluster_settings,
                     _choose_use_rep_as_scanpy,
                     _recreate_preprocessed_view,
                     _extract_valid_pca_kwargs)
from ._pca import _pca
from ._neighbors import _neighbors

from .._utils import (_default_gate_and_default_layer,
                      _enable_gate_aliases,
                      IMPLEMENTED_SCALERS)
from ..exceptions._exceptions import InvalidScalingError

@_default_gate_and_default_layer
@_enable_gate_aliases
def parc(adata: AnnData,
         gate: str,
         layer: str,
         key_added: Optional[str] = None,
         use_only_fluo: bool = True,
         exclude: Optional[Union[list[str], str]] = None,
         scaling: Optional[Literal["MinMaxScaler", "RobustScaler", "StandardScaler"]] = None,
         copy: bool = False,
         **kwargs) -> Optional[AnnData]:
    """\
    Computes PARC clustering. If PCA and neighbors have not been calculated for this
    gate and layer, the function will compute it automatically.

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
        annotations. Defaults to f'{gate}_{layer}_parc'.
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
        keyword arguments that are passed directly to the `parc.PARC`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.obs[f'{gate}_{layer}_parc]`
            cluster annotations
        `.uns['settings'][f'_parc_{gate}_{layer}]`
            Settings that were used for PARC calculation

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
    >>> fp.tl.parc(dataset)

    """
    
    adata = adata.copy() if copy else adata

    if exclude is None:
        exclude = []
    else:
        if not isinstance(exclude, list):
            exclude = [exclude]

    if scaling not in IMPLEMENTED_SCALERS and scaling is not None:
        raise InvalidScalingError(scaler = scaling)

    _save_cluster_settings(adata = adata,
                           gate = gate,
                           layer = layer,
                           use_only_fluo = use_only_fluo,
                           exclude = exclude,
                           scaling = scaling,
                           clustering = "parc",
                           **kwargs)

    uns_key = f"{gate}_{layer}"
    cluster_key = key_added or f"{uns_key}_parc"
    neighbors_key = f"{uns_key}_neighbors"
    connectivities_key = f"{neighbors_key}_connectivities"

    preprocessed_adata = _preprocess_adata(adata,
                                           gate = gate,
                                           layer = layer,
                                           use_only_fluo = use_only_fluo,
                                           exclude = exclude,
                                           scaling = scaling)

    if f"X_pca_{uns_key}" not in adata.obsm:
        print("computing PCA for parc!")
        pca_kwargs = _extract_valid_pca_kwargs(kwargs)
        adata = _pca(adata = adata,
                     preprocessed_adata = preprocessed_adata,
                     dimred_key = f"pca_{uns_key}",
                     **pca_kwargs)
        preprocessed_adata = _recreate_preprocessed_view(adata,
                                                         preprocessed_adata)

    if connectivities_key not in adata.obsp:
        print("computing neighbors for parc!")
        neighbors_kwargs = _extract_valid_neighbors_kwargs(kwargs)
        if not "use_rep" in neighbors_kwargs:
            neighbors_kwargs["use_rep"] = _choose_use_rep_as_scanpy(adata,
                                                                    uns_key = uns_key,
                                                                    use_rep = None,
                                                                    n_pcs = neighbors_kwargs.get("n_pcs"))
        adata = _neighbors(adata = adata,
                           preprocessed_adata = preprocessed_adata,
                           neighbors_key = neighbors_key,
                           **neighbors_kwargs)
        preprocessed_adata = _recreate_preprocessed_view(adata,
                                                         preprocessed_adata)

    parc_kwargs = _extract_valid_parc_kwargs(kwargs)
    parcer = _parc.PARC(preprocessed_adata.X,
                        neighbor_graph = preprocessed_adata.obsp[connectivities_key]
                                         if connectivities_key in adata.obsp else None,
                        **parc_kwargs)
    parcer.run_PARC()

    adata = _merge_cluster_info_into_adata(adata,
                                           preprocessed_adata,
                                           cluster_key = cluster_key,
                                           cluster_assignments = parcer.labels)
       
    return adata if copy else None

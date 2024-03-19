from anndata import AnnData
import numpy as np

from ._utils import _preprocess_adata, _merge_dimred_coordinates_into_adata

from ..exceptions._exceptions import ReductionNotFoundError
from .._utils import _default_gate_and_default_layer


@_default_gate_and_default_layer
def scanorama_integrate(
    adata: AnnData,
    gate: str,
    layer: str,
    key: str,
    basis: str = 'pca',
    adjusted_basis: str = 'scanorama',
    knn: int = 20,
    sigma: float = 15,
    approx: bool = True,
    alpha: float = 0.10,
    batch_size: int = 5000,
    copy: bool = False,
    **kwargs,
):
    """\
    Computes scanorama integration.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    gate
        The gate to be analyzed, called by the population name.
        This parameter has a default stored in fp.settings, but
        can be superseded by the user.
    layer
        The layer corresponding to the data matrix. Similar to the
        gate parameter, it has a default stored in fp.settings which
        can be overwritten by user input.
    key
        Column in `.obs` that specifies the batch.
    basis
        Entry in `.obsm` that specifies the embedding.
    adjusted_basis
        Name of the integrated embedding. 
    knn
        Number of nearest neighbors to use for matching.
    sigma
        Correction smoothing parameter on Gaussian kernel.
    approx
        Use approximate nearest neighbors with Python ``annoy``;
        greatly speeds up matching runtime.
    alpha
        Alignment score minimum cutoff.
    batch_size
        The batch size used in the alignment vector computation. Useful
        when integrating very large (>100k samples) datasets. Set to
        large value that runs within available memory.

        Return a copy of adata instead of modifying inplace.
    **kwargs : dict, optional
        keyword arguments that are passed directly to the `scanorama.scanorama`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.obsm[adjusted_basis]`
            integrated embedding of the data


    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset
    AnnData object with n_obs × n_vars = 615936 × 22
    obs: 'sample_ID', 'file_name', 'condition', 'sex', 'batch'
    var: 'pns', 'png', 'pne', 'pnr', 'type', 'pnn', 'cofactors'
    uns: 'metadata', 'panel', 'workspace', 'gating_cols', 'dataset_status_hash'
    obsm: 'gating'
    layers: 'compensated', 'transformed'
    >>> fp.settings.default_gate = "T_cells"
    >>> fp.settings.default_layer = "transformed"
    >>> fp.tl.pca(dataset)
    >>> fp.tl.scanorama_integrate(
    ...     dataset,
    ...     basis = "X_pca_T_cells_transformed",
    ...     adjusted_basis = "X_pca_T_cells_transformed_scanorama",
    ... )

    """


    adata = adata.copy() if copy else adata

    full_basis = f"X_{basis}_{gate}_{layer}"
    _adjusted_basis = f"{adjusted_basis}_{gate}_{layer}"

    # we preprocess to subset the gate as harmony does not allow NaN
    preprocessed_adata = _preprocess_adata(adata = adata,
                                           gate = gate,
                                           layer = layer)
                                           
    if full_basis not in adata.obsm:
        raise ReductionNotFoundError(basis)

    try:
        import scanorama
    except ImportError:
        raise ImportError("\nplease install Scanorama:\n\n\tpip install scanorama")

    # Get batch indices in linear time.
    curr_batch = None
    batch_names = []
    name2idx = {}
    for idx in range(preprocessed_adata.X.shape[0]):
        batch_name = preprocessed_adata.obs[key][idx]
        if batch_name != curr_batch:
            curr_batch = batch_name
            if batch_name in batch_names:
                # Contiguous batches important for preserving cell order.
                raise ValueError('Detected non-contiguous batches.')
            batch_names.append(batch_name)  # Preserve name order.
            name2idx[batch_name] = []
        name2idx[batch_name].append(idx)

    # Separate batches.
    datasets_dimred = [
        preprocessed_adata.obsm[full_basis][name2idx[batch_name]] for batch_name in batch_names
    ]

    # Integrate.
    integrated = scanorama.assemble(
        datasets_dimred,  # Assemble in low dimensional space.
        knn=knn,
        sigma=sigma,
        approx=approx,
        alpha=alpha,
        ds_names=batch_names,
        **kwargs,
    )

    integrated_embedding = np.concatenate(integrated)

    adata = _merge_dimred_coordinates_into_adata(adata,
                                                 preprocessed_adata,
                                                 coordinates = integrated_embedding,
                                                 dimred = basis,
                                                 dimred_key = _adjusted_basis)
    
    return adata if copy else None
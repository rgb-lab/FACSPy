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
    Use Scanorama [Hie19]_ to integrate different experiments.

    Scanorama [Hie19]_ is an algorithm for integrating single-cell
    data from multiple experiments stored in an AnnData object. This
    function should be run after performing PCA but before computing
    the neighbor graph, as illustrated in the example below.

    This uses the implementation of `scanorama
    <https://github.com/brianhie/scanorama>`__ [Hie19]_.

    Parameters
    ----------
    adata
        The annotated data matrix.
    key
        The name of the column in ``adata.obs`` that differentiates
        among experiments/batches. Cells from the same batch must be
        contiguously stored in ``adata``.
    basis
        The name of the field in ``adata.obsm`` where the PCA table is
        stored. Defaults to ``'X_pca'``, which is the default for
        ``sc.tl.pca()``.
    adjusted_basis
        The name of the field in ``adata.obsm`` where the integrated
        embeddings will be stored after running this function. Defaults
        to ``X_scanorama``.
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
    kwargs
        Any additional arguments will be passed to
        ``scanorama.integrate()``.

    Returns
    -------
    Updates adata with the field ``adata.obsm[adjusted_basis]``,
    containing Scanorama embeddings such that different experiments
    are integrated.

    Example
    -------
    First, load libraries and example dataset, and preprocess.

    >>> import scanpy as sc
    >>> import scanpy.external as sce
    >>> adata = sc.datasets.pbmc3k()
    >>> sc.pp.recipe_zheng17(adata)
    >>> sc.tl.pca(adata)

    We now arbitrarily assign a batch metadata variable to each cell
    for the sake of example, but during real usage there would already
    be a column in ``adata.obs`` giving the experiment each cell came
    from.

    >>> adata.obs['batch'] = 1350*['a'] + 1350*['b']

    Finally, run Scanorama. Afterwards, there will be a new table in
    ``adata.obsm`` containing the Scanorama embeddings.

    >>> sce.pp.scanorama_integrate(adata, 'batch')
    >>> 'X_scanorama' in adata.obsm
    True
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
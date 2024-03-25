from anndata import AnnData
import numpy as np
from typing import Optional, Union
import warnings
from typing import Literal
from anndata import AnnData

from scanpy._settings import settings
from ._dr_samplewise import _perform_samplewise_dr
from ._utils import (_choose_representation,
                     _merge_dimred_coordinates_into_adata,
                     _add_uns_data)
from .._utils import _default_layer

@_default_layer
def tsne_samplewise(adata: AnnData,
                    layer: str,
                    n_components: int = 3,
                    use_only_fluo: bool = True,
                    exclude: Optional[Union[list[str], str]] = None,
                    scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",
                    data_group: str = "sample_ID",
                    data_metric: Literal["mfi", "fop"] = "mfi",
                    copy: bool = False,
                    *args,
                    **kwargs) -> Optional[AnnData]:
    """\
    Computes samplewise TSNE based on either the median fluorescence values (MFI)
    or frequency of parent values (FOP). TSNE will be calculated for all gates at once.
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
        Parameter to specify if the TSNE should only be calculated for the fluorescence
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
        keyword arguments that are passed directly to the `sklearn.TSNE`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.uns[f'{data_metric}_{data_group}_{layer}']`
            TSNE coordinates are added to the respective frame
        `.uns['settings'][f"_tsne_samplewise_{data_metric}_{layer}"]`
            Settings that were used for samplewise TSNE calculation

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
    >>> fp.tl.tsne_samplewise(dataset)

    """

    adata = adata.copy() if copy else adata

    adata = _perform_samplewise_dr(adata = adata,
                                   reduction = "TSNE",
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

def _tsne(adata: AnnData,
          preprocessed_adata: AnnData,
          uns_key: str,
          dimred_key: str,
          **kwargs) -> AnnData:

    coords, tsne_params = _compute_tsne(preprocessed_adata,
                                        uns_key = uns_key,
                                        **kwargs)

    adata = _merge_dimred_coordinates_into_adata(adata = adata,
                                                 gate_subset = preprocessed_adata,
                                                 coordinates = coords,
                                                 dimred = "tsne",
                                                 dimred_key = dimred_key)
    
    adata = _add_uns_data(adata = adata,
                          data = tsne_params,
                          key_added = dimred_key)
    
    return adata

def _compute_tsne(adata: AnnData,
                  uns_key: str = None,
                  n_components: int = 3,
                  n_pcs: Optional[int] = None,
                  use_rep: Optional[str] = None,
                  perplexity: Union[float, int] = 30,
                  early_exaggeration: Union[float, int] = 12,
                  learning_rate: Union[float, int] = "auto",
                  random_state: int = 187,
                  use_fast_tsne: bool = False,
                  n_jobs: Optional[int] = None,
                  *,
                  metric: str = "euclidean") -> tuple[np.ndarray, dict]:
    """\
    Internal function to compute the TSNE embedding. The core of the function
    is implemented from scanpy with the important difference that the TSNE
    coordinates are returned and not written to the adata object.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    uns_key
        Name of the slot in `.obsm` that the TSNE is calculated on.
    n_components
        The number of dimensions of the embedding.
    n_pcs
        Number of principal components to use
    use_rep
        Name of the slot in `.obsm` that specifies the embedding
        that TSNE is calculated on. Used in conjunction with `uns_key`.
    perplexity
        The perplexity is related to the number of nearest neighbors
        that is used in other manifold learning algorithms. Larger
        datasets usually require a larger perplexity. Consider
        selecting a value between 5 and 50. Different values
        can result in significantly different results. The perplexity
        must be less than the number of samples.
    early_exaggeration
        Controls how tight natural clusters in the original space
        are in the embedded space and how much space will be between
        them. For larger values, the space between natural clusters
        will be larger in the embedded space. Again, the choice of
        this parameter is not very critical. If the cost function
        increases during initial optimization, the early exaggeration
        factor or the learning rate might be too high.
    learning_rate
        Learning rate. The learning rate for t-SNE is usually in the
        range [10.0, 1000.0]. If the learning rate is too high, the
        data may look like a 'ball' with any point approximately
        equidistant from its nearest neighbours. If the learning rate
        is too low, most points may look compressed in a dense cloud
        with few outliers. If the cost function gets stuck in a bad
        local minimum increasing the learning rate may help.
    random_state
        Sets the random state for the algorithm.
    use_fast_tsne
        Whether to use the fast_tsne implementation
    n_jobs
        number of CPU cores to use.

    Returns
    -------
    X_tsne
        The TSNE coordinates
    params
        A dictionary containing the parameters used for analysis.
    
    """

    X = _choose_representation(adata,
                               uns_key = uns_key,
                               use_rep = use_rep,
                               n_pcs = n_pcs)
    # params for sklearn
    n_jobs = settings.n_jobs if n_jobs is None else n_jobs
    params_sklearn = dict(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        verbose=settings.verbosity > 3,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_jobs=n_jobs,
        metric=metric,
    )
    # square_distances will default to true in the future, we'll get ahead of the
    # warning for now

    # This results in a unexpected keyword argument for TSNE...
#    if metric != "euclidean":
#        sklearn_version = version.parse(sklearn.__version__)
#        if sklearn_version >= version.parse("0.24.0"):
#            params_sklearn["square_distances"] = True
#        else:
#            warnings.warn(
#                "Results for non-euclidean metrics changed in sklearn 0.24.0, while "
#                f"you are using {sklearn.__version__}.",
#                UserWarning,
#            )

    # Backwards compat handling: Remove in scanpy 1.9.0
    if n_jobs != 1 and not use_fast_tsne:
        warnings.warn(
            UserWarning(
                "In previous versions of scanpy, calling tsne with n_jobs > 1 would use "
                "MulticoreTSNE. Now this uses the scikit-learn version of TSNE by default. "
                "If you'd like the old behaviour (which is deprecated), pass "
                "'use_fast_tsne=True'. Note, MulticoreTSNE is not actually faster anymore."
            )
        )
    if use_fast_tsne:
        warnings.warn(
            FutureWarning(
                "Argument `use_fast_tsne` is deprecated, and support for MulticoreTSNE "
                "will be dropped in a future version of scanpy."
            )
        )

    # deal with different tSNE implementations
    if use_fast_tsne:
        try:
            from MulticoreTSNE import MulticoreTSNE as TSNE

            tsne = TSNE(**params_sklearn)
            # need to transform to float64 for MulticoreTSNE...
            X_tsne = tsne.fit_transform(X.astype('float64'))
        except ImportError:
            use_fast_tsne = False
            warnings.warn(
                UserWarning(
                    "Could not import 'MulticoreTSNE'. Falling back to scikit-learn."
                )
            )
    if use_fast_tsne is False:  # In case MultiCore failed to import
        from sklearn.manifold import TSNE

        # unfortunately, sklearn does not allow to set a minimum number
        # of iterations for barnes-hut tSNE
        tsne = TSNE(**params_sklearn)
        X_tsne = tsne.fit_transform(X)

    params = {
        "params": {
            k: v
            for k, v in {
                "perplexity": perplexity,
                "early_exaggeration": early_exaggeration,
                "learning_rate": learning_rate,
                "n_jobs": n_jobs,
                "metric": metric,
                "use_rep": use_rep,
                "n_components": n_components
            }.items()
            if v is not None
        }
    }
    return X_tsne, params

from anndata import AnnData
from typing import Optional, Union
import warnings
from typing import Literal
from anndata import AnnData

from ._dr_samplewise import _perform_samplewise_dr
from ._utils import (_choose_representation,
                     _merge_dimred_coordinates_into_adata,
                     _add_uns_data)
from scanpy._settings import settings
from .._utils import _default_layer

@_default_layer
def tsne_samplewise(adata: AnnData,
                    data_group: Optional[Union[str, list[str]]] = "sample_ID",
                    data_metric: Literal["mfi", "fop", "gate_frequency"] = "mfi",
                    layer: str = None,
                    use_only_fluo: bool = True,
                    exclude: Optional[Union[str, list, str]] = None,
                    scaling: Literal["MinMaxScaler", "RobustScaler", "StandardScaler"] = "MinMaxScaler",
                    n_components: int = 3,
                    copy = False,
                    *args,
                    **kwargs) -> Optional[AnnData]:

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
                  learning_rate: Union[float, int] = 1000,
                  random_state: int = 187,
                  use_fast_tsne: bool = False,
                  n_jobs: Optional[int] = None,
                  *,
                  metric: str = "euclidean") -> Optional[AnnData]:
    """\
    t-SNE [Maaten08]_ [Amir13]_ [Pedregosa11]_.

    t-distributed stochastic neighborhood embedding (tSNE) [Maaten08]_ has been
    proposed for visualizating single-cell data by [Amir13]_. Here, by default,
    we use the implementation of *scikit-learn* [Pedregosa11]_. You can achieve
    a huge speedup and better convergence if you install `Multicore-tSNE
    <https://github.com/DmitryUlyanov/Multicore-TSNE>`__ by [Ulyanov16]_, which
    will be automatically detected by Scanpy.

    Parameters
    ----------
    adata
        Annotated data matrix.
    {doc_n_pcs}
    {use_rep}
    perplexity
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. The choice is not extremely critical since t-SNE
        is quite insensitive to this parameter.
    metric
        Distance metric calculate neighbors on.
    early_exaggeration
        Controls how tight natural clusters in the original space are in the
        embedded space and how much space will be between them. For larger
        values, the space between natural clusters will be larger in the
        embedded space. Again, the choice of this parameter is not very
        critical. If the cost function increases during initial optimization,
        the early exaggeration factor or the learning rate might be too high.
    learning_rate
        Note that the R-package "Rtsne" uses a default of 200.
        The learning rate can be a critical parameter. It should be
        between 100 and 1000. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high. If the cost function gets stuck in a bad local
        minimum increasing the learning rate helps sometimes.
    random_state
        Change this to use different intial states for the optimization.
        If `None`, the initial state is not reproducible.
    n_jobs
        Number of jobs for parallel computation.
        `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
    copy
        Return a copy instead of writing to `adata`.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

    **X_tsne** : `np.ndarray` (`adata.obs`, dtype `float`)
        tSNE coordinates of data.
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
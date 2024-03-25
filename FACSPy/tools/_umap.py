from anndata import AnnData
from typing import Optional, Literal

import warnings
from typing import Optional, Union
import warnings

import numpy as np
from packaging import version
from anndata import AnnData
from sklearn.utils import check_random_state, check_array

from scanpy.tools._utils import get_init_pos_from_paga
from scanpy._settings import settings
from scanpy._utils import AnyRandom, NeighborsView

from ._dr_samplewise import _perform_samplewise_dr
from ._utils import (_merge_dimred_coordinates_into_adata,
                     _add_uns_data,
                     _choose_representation)
from .._utils import _default_layer
_InitPos = Literal['paga', 'spectral', 'random']

@_default_layer
def umap_samplewise(adata: AnnData,
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
    Computes samplewise UMPA based on either the median fluorescence values (MFI)
    or frequency of parent values (FOP). UMAP will be calculated for all gates at once.
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
        Parameter to specify if the UMAP should only be calculated for the fluorescence
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
        keyword arguments that are passed directly to the `umap.UMAP`
        function. Please refer to its documentation.
    
    Returns
    -------
    :class:`~anndata.AnnData` or None
        Returns adata if `copy = True`, otherwise adds fields to the anndata
        object:

        `.uns[f'{data_metric}_{data_group}_{layer}']`
            UMAP coordinates are added to the respective frame
        `.uns['settings'][f"_umap_samplewise_{data_metric}_{layer}"]`
            Settings that were used for samplewise UMAP calculation

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
    >>> fp.tl.umap_samplewise(dataset)

    """

    adata = adata.copy() if copy else adata

    adata = _perform_samplewise_dr(adata = adata,
                                   reduction = "UMAP",
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


def _umap(adata: AnnData,
          preprocessed_adata: AnnData,
          neighbors_key: str,
          dimred_key: str,
          uns_key: str,
          **kwargs) -> AnnData:
    coords, umap_params = _compute_umap(preprocessed_adata,
                                        uns_key = uns_key,
                                        neighbors_key = neighbors_key,
                                        **kwargs)
    
    adata = _merge_dimred_coordinates_into_adata(adata = adata,
                                                 gate_subset = preprocessed_adata,
                                                 coordinates = coords,
                                                 dimred = "umap",
                                                 dimred_key = dimred_key)
    
    adata = _add_uns_data(adata = adata,
                          data = umap_params,
                          key_added = dimred_key)
    
    return adata

def _compute_umap(adata: AnnData,
                  uns_key: str,
                  min_dist: float = 0.5,
                  spread: float = 1.0,
                  n_components: int = 3,
                  maxiter: Optional[int] = None,
                  alpha: float = 1.0,
                  gamma: float = 1.0,
                  negative_sample_rate: int = 5,
                  init_pos: Union[_InitPos, np.ndarray, None] = 'spectral',
                  random_state: AnyRandom = 187,
                  a: Optional[float] = None,
                  b: Optional[float] = None,
                  method: Literal['umap', 'rapids'] = 'umap',
                  neighbors_key: Optional[str] = None) -> tuple[np.ndarray, dict]:
    """\
    Internal function to compute the UMAP embedding. The core of the function
    is implemented from scanpy with the important difference that the UMAP
    coordinates are returned and not written to the adata object.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels.
    uns_key
        Name of the slot in `.obsm` that the UMAP is calculated on.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out. The default of in the `umap-learn` package is
        0.1.
    spread
        The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called `n_epochs`
        in the original UMAP.
    alpha
        The initial learning rate for the embedding optimization.
    gamma
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Called `init` in the
        original UMAP. Options are:

        * Any key for `adata.obsm`.
        * 'paga': positions from :func:`~scanpy.pl.paga`.
        * 'spectral': use a spectral embedding of the graph.
        * 'random': assign initial embedding positions at random.
        * A numpy array of initial embedding positions.
    random_state
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` or `Generator`, `random_state` is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    a
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    b
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    method
        Use the original 'umap' implementation, or 'rapids' (experimental, GPU only)
    neighbors_key
        If specified, umap looks .uns[neighbors_key] for neighbors settings and
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.

    Returns
    -------
    X_umap
        The UMAP coordinates
    params
        A dictionary containing the parameters used for analysis.
    """

    if neighbors_key is None:
        raise ValueError("neighbors key has to be supplied!")

    neighbors = NeighborsView(adata, neighbors_key)

    # Compat for umap 0.4 -> 0.5
    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        import umap

    if version.parse(umap.__version__) >= version.parse("0.5.0"):

        def simplicial_set_embedding(*args, **kwargs):
            from umap.umap_ import simplicial_set_embedding

            X_umap, _ = simplicial_set_embedding(
                *args,
                densmap=False,
                densmap_kwds={},
                output_dens=False,
                **kwargs,
            )
            return X_umap

    else:
        from umap.umap_ import simplicial_set_embedding
    from umap.umap_ import find_ab_params

    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    else:
        a = a
        b = b
    if isinstance(init_pos, str) and init_pos in adata.obsm.keys():
        init_coords = adata.obsm[init_pos]
    elif isinstance(init_pos, str) and init_pos == 'paga':
        init_coords = get_init_pos_from_paga(
            adata, random_state=random_state, neighbors_key=neighbors_key
        )
    else:
        init_coords = init_pos  # Let umap handle it
    if hasattr(init_coords, "dtype"):
        init_coords = check_array(init_coords, dtype=np.float32, accept_sparse=False)

    random_state = check_random_state(random_state)

    neigh_params = neighbors['params']
    X = _choose_representation(
        adata,
        uns_key = uns_key,
        use_rep = neigh_params.get('use_rep', None),
        n_pcs = neigh_params.get('n_pcs', None),
    )
    if method == 'umap':
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding
        default_epochs = 500 if neighbors['connectivities'].shape[0] <= 10000 else 200
        n_epochs = default_epochs if maxiter is None else maxiter
        X_umap = simplicial_set_embedding(
            X,
            neighbors['connectivities'].tocoo(),
            n_components,
            alpha,
            a,
            b,
            gamma,
            negative_sample_rate,
            n_epochs,
            init_coords,
            random_state,
            neigh_params.get('metric', 'euclidean'),
            neigh_params.get('metric_kwds', {}),
            verbose=settings.verbosity > 3,
        )
    elif method == 'rapids':
        metric = neigh_params.get('metric', 'euclidean')
        if metric != 'euclidean':
            raise ValueError(
                f'`sc.pp.neighbors` was called with `metric` {metric!r}, '
                "but umap `method` 'rapids' only supports the 'euclidean' metric."
            )
        from cuml import UMAP

        n_neighbors = neighbors['params']['n_neighbors']
        n_epochs = (
            500 if maxiter is None else maxiter
        )  # 0 is not a valid value for rapids, unlike original umap
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        umap = UMAP(
            n_neighbors = n_neighbors,
            n_components = n_components,
            n_epochs = n_epochs,
            learning_rate = alpha,
            init = init_pos,
            min_dist = min_dist,
            spread = spread,
            negative_sample_rate = negative_sample_rate,
            a = a,
            b = b,
            verbose = settings.verbosity > 3,
            random_state = random_state,
        )
        X_umap = umap.fit_transform(X_contiguous)
    
    params = {'params': {'a': a, 'b': b}}
    if random_state != 0:
        params['random_state'] = random_state

    return X_umap, params

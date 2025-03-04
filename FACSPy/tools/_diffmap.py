from scanpy.tools._dpt import DPT
from anndata import AnnData
import numpy as np
from typing import Optional

from ._utils import _merge_dimred_coordinates_into_adata, _add_uns_data

def _diffmap(adata: AnnData,
             preprocessed_adata: AnnData,
             neighbors_key: str,
             dimred_key: str,
             uns_key: str,
             **kwargs) -> AnnData:

    coords, diffmap_evals = _compute_diffmap(adata = preprocessed_adata,
                                             neighbors_key = neighbors_key,
                                             **kwargs)


    adata = _merge_dimred_coordinates_into_adata(adata = adata,
                                                 gate_subset = preprocessed_adata,
                                                 coordinates = coords,
                                                 dimred = "diffmap",
                                                 dimred_key = dimred_key)
    
    adata = _add_uns_data(adata = adata,
                          data = diffmap_evals,
                          key_added = f"{uns_key}_diffmap_evals")
    
    return adata

def _compute_diffmap(adata: AnnData,
                     n_comps: int = 15,
                     neighbors_key: Optional[str] = None,
                     random_state: int = 187) -> tuple[np.ndarray, np.ndarray]:
    """\
    Internal function to compute the Diffmap embedding. The core of the function
    is implemented from scanpy with the important difference that the diffmap
    coordinates are returned and not written to the adata object.

    Parameters
    ----------
    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    n_comps
        The number of dimensions of the representation.
    neighbors_key
        If specified, umap looks .uns[neighbors_key] for neighbors settings and
        .obsp[f'{neighbors_key}']['connectivities_key'] for connectivities.

    Returns
    -------
    dpt.eigen_basis
        The diffusion map coordinates
    dpt.eigen_values
        The diffusion map eigen values
    """

    dpt = DPT(adata,
              neighbors_key = neighbors_key)
    dpt.compute_transitions()
    dpt.compute_eigen(n_comps = n_comps,
                      random_state = random_state)
    return dpt.eigen_basis, dpt.eigen_values
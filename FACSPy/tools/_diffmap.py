from scanpy.tools._dpt import DPT
from anndata import AnnData
import numpy as np
from typing import Optional

from ._utils import _merge_dimred_coordinates_into_adata, _add_uns_data

def _diffmap(adata: AnnData,
             preprocessed_adata: AnnData,
             neighbors_key: str,
             dimred_key: str,
             uns_key: str) -> AnnData:

    coords, diffmap_evals = _compute_diffmap(adata = preprocessed_adata,
                                             n_comps = 3,
                                             neighbors_key = neighbors_key)

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
    # This module copies the sc.tl.diffmap function
    # with the important difference that nothing 
    # gets written to the dataset directly. That way,
    # we keep the anndatas as Views when multiple gates are
    # analyzed
    dpt = DPT(adata,
              neighbors_key = neighbors_key)
    dpt.compute_transitions()
    dpt.compute_eigen(n_comps = n_comps,
                      random_state = random_state)
    return dpt.eigen_basis, dpt.eigen_values
    
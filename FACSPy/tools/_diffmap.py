from scanpy.tools._dpt import DPT
from anndata import AnnData
import numpy as np

def _compute_diffmap(adata: AnnData,
                     n_comps=15,
                     neighbors_key = None,
                     random_state = 0) -> tuple[np.ndarray, np.ndarray]:
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
    
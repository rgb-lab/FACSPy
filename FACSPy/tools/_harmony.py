from anndata import AnnData

from ._utils import _preprocess_adata, _merge_dimred_coordinates_into_adata

from ..exceptions._exceptions import ReductionNotFoundError
from .._utils import _default_gate_and_default_layer

@_default_gate_and_default_layer
def harmony_integrate(adata: AnnData,
                      gate: str,
                      layer: str,
                      key: str,
                      basis: str = "pca",
                      adjusted_basis: str = "pca_harmony",
                      copy: bool = False,
                      **kwargs):
    
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
        import harmonypy
    except ImportError:
        raise ImportError("\nplease install harmonypy:\n\n\tpip install harmonypy")
    
    harmony_out = harmonypy.run_harmony(preprocessed_adata.obsm[full_basis],
                                        preprocessed_adata.obs,
                                        key,
                                        **kwargs)

    integrated_embedding = harmony_out.Z_corr.T

    adata = _merge_dimred_coordinates_into_adata(adata,
                                                 preprocessed_adata,
                                                 coordinates = integrated_embedding,
                                                 dimred = basis,
                                                 dimred_key = _adjusted_basis)

    return adata if copy else None
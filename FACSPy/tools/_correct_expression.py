from anndata import AnnData
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from .._utils import _fetch_fluo_channels, subset_fluo_channels

from typing import Optional

def correct_expression(adata: AnnData,
                       layer: str = "compensated",
                       use_only_fluo: bool = True,
                       batch_key: str = "batch",
                       embedding_key: str = "X_pca",
                       reference_samples: Optional[list[str]] = None,
                       reference_key: Optional[str] = None,
                       reference_value: str = None,
                       key_added: str = "integrated",
                       n_comps: int = 20,
                       # iterations: int = 1,
                       return_matrix: bool = False,
                       copy: bool = False
                       ):
    """
    Function to correct the expression values based on a joint PCA embedding.

    Parameters
    ----------
    embedding_key
        key in adata.obsm that stores the integrated embedding
    reference_samples
        a list of sample_IDs corresponding to the reference samples. Will
        be combined the batch_key information to extract reference samples
    reference_key
        Key of the .obs slot column that points to the reference samples
    reference_value
        specifies the value which marks the reference samples, e.g. "ref".
        Other values will be ignored and treated as non-reference samples
    key_added
        Key in adata.layers that stores the corrected expression matrix
    iterations
        repeat the PCA correction step n times

    Returns
    -------

    If copy, the anndata instance with new layers
    """

    adata = adata.copy() if copy else adata
    integrated_embedding = _extract_integrated_embedding_from_adata(adata,
                                                                    embedding_key)
    layer_copy = adata.layers[layer].copy()
                                                               
    if n_comps > integrated_embedding.shape[1]:
        print(f"Warning! n_comps set to {integrated_embedding.shape[1]}")
        n_comps = integrated_embedding.shape[1]

    df = adata.to_df(layer = layer).copy()

    if reference_samples is None and reference_key is None:
        # no reference sample present, so we correct for every sample
        indices = adata.obs_names
        expr_data = _extract_expr_data_from_adata(adata,
                                                  layer = layer,
                                                  use_only_fluo = use_only_fluo)
        pca_ = _fit_pca(expr_data)
        corrected_matrix = _get_corrected_matrix(pca_,
                                                 expr_data = expr_data,
                                                 integrated_embedding = integrated_embedding,
                                                 n_comps = n_comps)
        df = _merge_corrected_matrix_into_df(adata,
                                             corrected_matrix,
                                             df,
                                             indices,
                                             use_only_fluo)
    else:
        if reference_key is not None:
            print("extracting reference sample IDs from adata")
            ref_sample_IDs = _get_ref_sample_IDs(adata,
                                                 reference_key,
                                                 reference_value)
        else:
            print(f"using {reference_samples} as references")
            ref_sample_IDs = reference_samples

        assert _every_batch_has_ref_sample()

        for batch in adata.obs[batch_key].unique():
            batch_samples = adata.obs.loc[adata.obs[batch_key] == batch, "sample_IDs"].unique()
            ref_sample = [sample for sample in ref_sample_IDs if sample in batch_samples]

            batch_data = adata[adata.obs[batch_key] == batch,:]
            indices = batch_data.obs_names
            ref_data = adata[adata.obs["sample_ID"] == ref_sample,:]

            ref_expr_data = _extract_expr_data_from_adata(ref_data,
                                                          layer = layer,
                                                          use_only_fluo = use_only_fluo)
            ref_pca_ = _fit_pca(ref_expr_data)

            expr_data = _extract_expr_data_from_adata(batch_data,
                                                      layer = layer,
                                                      use_only_fluo = use_only_fluo)

            batch_specific_corrected_matrix = _get_corrected_matrix(pca_ = ref_pca_,
                                                                    expr_data = expr_data)

            ## store it somewhere...
            df = _merge_corrected_matrix_into_df(adata,
                                                 batch_specific_corrected_matrix,
                                                 df,
                                                 indices,
                                                 use_only_fluo)
    assert np.array_equal(adata.layers[layer], layer_copy)

    corrected_matrix = df.values
    assert np.array_equal(adata.layers[layer], layer_copy)
    if return_matrix:
        return corrected_matrix

    adata = _merge_corrected_matrix_into_adata(adata = adata,
                                               layer = layer,
                                               corrected_matrix = corrected_matrix,
                                               key_added = key_added,
                                               use_only_fluo = use_only_fluo)

    return adata if copy else None

def _merge_corrected_matrix_into_df(adata:AnnData,
                                    matrix: np.ndarray,
                                    df: pd.DataFrame,
                                    indices: pd.Index,
                                    use_only_fluo: bool):
    if use_only_fluo:
        fluo_channels = _fetch_fluo_channels(adata)
        df.loc[indices,fluo_channels] = matrix
    else:
        df.loc[indices,:] = matrix
    return df


def _every_batch_has_ref_sample(adata: AnnData,
                                ref_samples: list[str],
                                batch_key: str):
    """ TODO! """
    return True

def _get_ref_sample_IDs(adata: AnnData,
                        reference_key: str,
                        reference_value: str):
    return adata.obs.loc[adata.obs[reference_key] == reference_value, "sample_ID"].unique().tolist()

def _merge_corrected_matrix_into_adata(adata: AnnData,
                                       layer: str,
                                       corrected_matrix: np.ndarray,
                                       key_added: str,
                                       use_only_fluo: bool):
    if use_only_fluo:
        fluo_channels = _fetch_fluo_channels(adata)
        df = adata.to_df(layer = layer)
        df.loc[fluo_channels] = corrected_matrix
        adata.layers[key_added] = df.values
    else:
        adata.layers[key_added] = corrected_matrix
    return adata

def _extract_integrated_embedding_from_adata(adata: AnnData,
                                             embedding_key: str) -> np.ndarray:
    return adata.obsm[embedding_key].copy()

def _extract_expr_data_from_adata(adata: AnnData,
                                  layer: str,
                                  use_only_fluo: bool):
    if use_only_fluo:
        adata = subset_fluo_channels(adata, as_view = True, copy = False)
        data = adata.layers[layer].copy()
    else:
        data = adata.layers[layer].copy()

    return data

def _fit_pca(expr_data) -> PCA:
    return PCA().fit(expr_data)

def _get_corrected_matrix(pca_: PCA,
                          expr_data: np.ndarray,
                          integrated_embedding: np.ndarray,
                          n_comps: int):
    mu = np.mean(expr_data, axis = 0)
    Xhat = np.dot(integrated_embedding[:,:n_comps], pca_.components_[:n_comps,:])
    Xhat += mu
    return Xhat

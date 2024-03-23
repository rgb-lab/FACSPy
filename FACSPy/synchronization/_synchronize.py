import warnings
from anndata import AnnData
from typing import Optional

from ._hash_generation import _generate_hash_dict
from ._sample_sync import (_sync_sample_ids_from_obs,
                           _sync_sample_ids_from_metadata,
                           _sync_columns_from_obs,
                           _sync_columns_from_metadata)
from ._var_sync import (_sync_cofactors_from_var,
                        _sync_panel_from_var,
                        _sync_var_from_panel,
                        _sync_cofactors_from_panel)
from ._utils import _sync_uns_frames

from ..exceptions._exceptions import DataModificationWarning, ModificationAmbiguityError

def _hash_dataset(adata: AnnData,
                          copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    adata.uns["dataset_status_hash"] = _generate_hash_dict(adata)
    return adata if copy else None

def _dataset_has_been_modified(adata: AnnData) -> bool:
    """
    Returns a boolean if the dataset has been modified.
    Calculates the hashs on the input dataset and compares
    it to the stored hashs

    Returns:
        bool: Answer if the dataset has been modified
    """
    return _generate_hash_dict(adata) != adata.uns["dataset_status_hash"]

def _get_modified_entities(adata) -> list[str]:
    current_hash_dict = adata.uns["dataset_status_hash"]
    comparison_hash_dict = _generate_hash_dict(adata)
    return [
        key
        for key in current_hash_dict
        if current_hash_dict[key] != comparison_hash_dict[key]
    ]

def synchronize_dataset(adata: AnnData,
                        recalculate: bool = False,
                        copy: bool = False
                        ) -> Optional[AnnData]:
    """
    This function is used to synchronize the unstructured metadata 
    with the underlying data.
    That way, we attempt to update the unstructured data whenever the 
    data are subset or changed in any way.

    To detect a changed dataset, we generate a hash that is based on the
    individual entries that we want to compare. If this hash changes, the dataset
    has been modified in some way.

    This function will only trigger modifications if the hash is not identical.
    That way, we save lots of unnecessary lookups.

    The hash is generated upon dataset creation and stored in adata.uns["dataset_status_hashs"].
    adata.uns["dataset_status_hashs"] is a dictionary where multiple entries can
    be inserted.

    Currently, `adata.obs_names`, `adata.var_names`, unique `sample_ID` in 
    `.obs` and in `.uns["metadata"] as well as the columns in `.obs` and in 
    `.uns["metadata"]` are hashed.

    Parameters
    ----------

    adata
        The anndata object of shape `n_obs` x `n_vars`
        where rows correspond to cells and columns to the channels
    recalculate
        If True, recalculates data stored in `adata.uns` based on the settings as stored
        in `adata.uns["settings"]`
    copy
        Whether to copy the dataset.

    Returns
    -------
    :class:`~anndata.AnnData` or None, depending on `copy`.

    Examples
    --------

    >>> import FACSPy as fp
    >>> dataset = fp.create_dataset(...)
    >>> dataset = dataset[dataset.obs["sample_ID"].isin(["1", "2"]),:]
    >>> fp.sync.synchronize_dataset(dataset)

    
    """
    adata = adata.copy() if copy else adata
    
    if not _dataset_has_been_modified(adata):
        print("dataset is already synchronized")
        return adata if copy else None
    
    modified_subsets = _get_modified_entities(adata)
    print(f"Found modified subsets: {modified_subsets}")

    # we issue a warning that the calculated values
    # are potentially wrong now and should be recalculated
    warnings.warn('', DataModificationWarning)

    if "adata_obs_names" in modified_subsets:

        # if obs_names have been modified we have two possibilities
        # 1) cells have been appended
        # 2) cells have been removed
        # Note: Cell shuffling does not change the hash

        if "adata_sample_ids" in modified_subsets:

            if "metadata_sample_ids" in modified_subsets:
                # if both the obs sampleIDs and the metadata sample_IDs
                # changed we dont know which one to believe. We raise
                # an error and point the user to the right function.
                raise ModificationAmbiguityError("sample_id", "obs", "metadata")

            # we check if whole samples were affected.
            # if samples are removed or appended, we have to
            # adjust the metadata.
            _sync_sample_ids_from_obs(adata)
            #_reset_hash()
        
    if "metadata_sample_ids" in modified_subsets:
        # we check if samples were deleted from the metadata.
        # If so, they will be removed from the dataset
        _sync_sample_ids_from_metadata(adata)
        
    if "adata_obs_columns" in modified_subsets:
        _sync_columns_from_obs(adata)

    if "metadata_columns" in modified_subsets:
        _sync_columns_from_metadata(adata)
    
    if "adata_var_names" in modified_subsets:
        # if the var names changed, we have to adjust the panel
        # the cofactor table and the uns frames
        _sync_panel_from_var(adata)
        _sync_cofactors_from_var(adata)

    if "panel_var_names" in modified_subsets:
        _sync_var_from_panel(adata)
        _sync_cofactors_from_panel(adata)

    _sync_uns_frames(adata,
                     recalculate = recalculate)

    _hash_dataset(adata)

    return adata if copy else None

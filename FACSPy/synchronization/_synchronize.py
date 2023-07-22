from anndata import AnnData
from typing import Union, Optional
from ._obs_sync import synchronize_samples
from ._var_sync import synchronize_vars

from ._hash_generation import generate_hash_dict

def hash_dataset(adata: AnnData,
                          copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    adata.uns["dataset_status_hash"] = generate_hash_dict(adata)
    return adata if copy else None

def dataset_has_been_modified(adata: AnnData) -> bool:
    """
    Returns a boolean if the dataset has been modified.
    Calculates the hashs on the input dataset and compares
    it to the stored hashs

    Returns:
        bool: Answer if the dataset has been modified
    """
    return generate_hash_dict(adata) != adata.uns["dataset_status_hash"]

def get_modified_entities(adata) -> list[str]:
    current_hash_dict = adata.uns["dataset_status_hash"]
    comparison_hash_dict = generate_hash_dict(adata)
    return [
        key
        for key in current_hash_dict
        if current_hash_dict[key] != comparison_hash_dict[key]
    ]

def synchronize_dataset(adata: AnnData,
                        copy: bool = False) -> Optional[AnnData]:
    """
    This function is used to synchronize the unstructured metadata 
    with the underlying data.
    That way, we attempt to update the unstructured data whenever the 
    data are subset or changed in any way.

    To detect a changed dataset, we generate a hash that is based on the
    individual things that we want to compare. If this hash changes, the dataset
    has been modified in any way.

    This function will only trigger modifications if the hash is not identical.
    That way, we save lots of unnecessary lookups.

    The hash is generated upon dataset creation and stored in adata.uns["dataset_status_hashs"].
    adata.uns["dataset_status_hashs"] is a dictionary where multiple entries can
    be inserted.


    """
    adata = adata.copy() if copy else adata
    
    if not dataset_has_been_modified(adata):
        print("dataset is already synchronized")
        return adata if copy else None
    
    modified_subsets = get_modified_entities(adata)
    print(modified_subsets)

    if "obs_names" in modified_subsets:
        synchronize_samples(adata)
    
    if "var_names" in modified_subsets:
        print("synchronizing vars")
        synchronize_vars(adata)

    hash_dataset(adata)




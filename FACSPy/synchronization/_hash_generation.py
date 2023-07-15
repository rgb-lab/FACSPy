from anndata import AnnData

def generate_hash_dict(adata: AnnData):
    """
    Stores all wanted hashs in a dictionary.
    This function can be appended to the liking.

    Args:
        adata (AnnData): _description_

    Returns:
        dictionary that contains all different hashs
    """
    return {
        "obs_names": generate_dataset_obs_hash(adata),
        "var_names": generate_dataset_var_hash(adata),
    }

def generate_dataset_obs_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the obs_names of the dataset

    Returns:
        int: the dataset obs hash
    """
    return hash(tuple(adata.obs_names.to_list()))

def generate_dataset_var_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the var_names of the dataset

    Returns:
        int: the dataset var hash
    """
    return hash(tuple(adata.var_names.to_list()))

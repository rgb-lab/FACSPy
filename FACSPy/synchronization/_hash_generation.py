from anndata import AnnData
import pandas as pd


def _generate_hash_dict(adata: AnnData):
    """
    Stores all wanted hashs in a dictionary.
    This function can be appended to the liking.

    Args:
        adata (AnnData): _description_

    Returns:
        dictionary that contains all different hashs
    """
    return {
        "adata_obs_names": HASH_FUNCTION_DICT["adata_obs_names"](adata),
        "adata_sample_ids": HASH_FUNCTION_DICT["adata_sample_ids"](adata),
        "adata_obs_columns": HASH_FUNCTION_DICT["adata_obs_columns"](adata),

        "metadata_sample_ids": HASH_FUNCTION_DICT["metadata_sample_ids"](adata),
        "metadata_columns": HASH_FUNCTION_DICT["metadata_columns"](adata),

        "adata_var_names": HASH_FUNCTION_DICT["adata_var_names"](adata),
        "panel_var_names": HASH_FUNCTION_DICT["panel_var_names"](adata)
    }

def _generate_dataset_obs_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the obs_names of the dataset

    Returns:
        int: the dataset obs hash
    """
    obs_names = adata.obs_names.sort_values(ascending = True)
    return hash(tuple(obs_names))


def _generate_panel_var_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the unique, sorted var_names of the panel

    Returns:
        int: dataset sample_ID hash
    
    """
    panel: pd.DataFrame = adata.uns["panel"].dataframe
    var_names = panel["fcs_colname"].sort_values(ascending = True)
    return hash(tuple(var_names))

def _generate_dataset_var_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the var_names of the dataset

    Returns:
        int: the dataset var hash
    """
    var_names = adata.var_names.sort_values(ascending = True)
    return hash(tuple(var_names))

def _generate_obs_sample_ID_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the unique, sorted sample_IDs

    Returns:
        int: dataset sample_ID hash
    
    """
    obs = adata.obs
    sample_ids = obs["sample_ID"].sort_values(ascending = True).unique()
    return hash(tuple(sample_ids))

def _generate_metadata_sample_ID_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the unique, sorted sample_IDs of the metadata

    Returns:
        int: dataset sample_ID hash
    
    """
    metadata: pd.DataFrame = adata.uns["metadata"].dataframe
    sample_ids = metadata["sample_ID"].sort_values(ascending = True).unique()
    return hash(tuple(sample_ids))

def _generate_obs_columns_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the unique, unsorted columns of .obs

    Returns:
        int: dataset sample_ID hash
    
    """
    obs_columns= adata.obs.columns
    return hash(tuple(obs_columns))

def _generate_metadata_columns_hash(adata: AnnData) -> int:
    """
    Creates a hash based on the unique, unsorted columns of the metadata

    Returns:
        int: dataset sample_ID hash
    
    """
    metadata: pd.DataFrame = adata.uns["metadata"].dataframe
    metadata_columns = metadata.columns
    return hash(tuple(metadata_columns))

HASH_FUNCTION_DICT = {
        "adata_obs_names": _generate_dataset_obs_hash,
        "adata_sample_ids": _generate_obs_sample_ID_hash,
        "adata_obs_columns": _generate_obs_columns_hash,

        "metadata_sample_ids": _generate_metadata_sample_ID_hash,
        "metadata_columns": _generate_metadata_sample_ID_hash,

        "adata_var_names": _generate_dataset_var_hash,
        "panel_var_names": _generate_panel_var_hash
}

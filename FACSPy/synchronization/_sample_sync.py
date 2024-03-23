from anndata import AnnData
import pandas as pd

from ._utils import _reset_hash
from ..dataset._supplements import Metadata

def _is_sample_specific_column(df: pd.DataFrame,
                               column: str,
                               n_sample_ids: int) -> bool:
    """
    Function that specifies if there is only one value
    per sample_ID in the dataframe
    """
    if column == "sample_ID":
        return True
    value_counts = df[["sample_ID", column]].value_counts()
    return value_counts.shape[0] == n_sample_ids

def _sync_columns_from_metadata(adata: AnnData) -> None:
    print("\t... synchronizing dataset to contain columns of the metadata object")
    metadata: pd.DataFrame = adata.uns["metadata"].dataframe
    metadata_columns = metadata.columns
    columns_to_transfer = [
        col for col in metadata_columns 
        if not col in adata.obs.columns
    ] + ["sample_ID"]
    original_idxs = adata.obs.index.tolist()
    obs = adata.obs.reset_index(names = "OBS_INDEX")

    new_frame = pd.merge(
        metadata[columns_to_transfer],
        obs,
        on = "sample_ID",
        how = "outer"
    )

    new_frame = new_frame.set_index("OBS_INDEX")
    new_frame = new_frame.loc[original_idxs,:]
    adata.obs = _sanitize_categoricals(new_frame)
    _reset_hash(adata, "adata_obs_columns")
    return

def _sync_columns_from_obs(adata: AnnData):
    print("\t... synchronizing metadata object to contain columns of the dataset")
    # first find the columns that have one value per sample_ID
    n_sample_ids = len(adata.obs["sample_ID"].unique())
    sample_specific_columns = [col for col in adata.obs.columns
                               if _is_sample_specific_column(adata.obs, col, n_sample_ids)]

    metadata: pd.DataFrame = adata.uns["metadata"].dataframe
    metadata_columns = metadata.columns
    columns_to_transfer = [col for col in sample_specific_columns
                           if not col in metadata_columns] + ["sample_ID"]

    metadata_to_append = adata.obs[columns_to_transfer].drop_duplicates()
    appended_metadata = pd.merge(
        metadata,
        metadata_to_append,
        on = "sample_ID",
        how = "outer"
    )
    adata.uns["metadata"] = Metadata(metadata = appended_metadata)
    _reset_hash(adata, "metadata_columns")
    return

def _sync_sample_ids_from_metadata(adata: AnnData):
    """\
    This function synchronizes the dataset by selecting the present sample_IDs
    in `adata.uns["metadata"]` and subsetting the dataset accordingly.

    Parameters
    ----------
    adata: AnnData
        the current dataset with an uns dict
    

    Examples
    --------
    >>> dataset = fp.create_dataset(...)
    >>> dataset.uns["metadata"].subset("sample_ID", ["1", "2", "3"])
    >>> fp.sync.sample_ids_from_metadata_to_obs(dataset)


    """
    print("\t... synchronizing dataset to contain sample_IDs of the metadata object")
    metadata_frame: pd.DataFrame = adata.uns["metadata"].dataframe.copy()
    obs = adata.obs

    metadata_sample_ids = metadata_frame["sample_ID"].tolist()
    obs_sample_ids = obs["sample_ID"].tolist()

    appended_sample_ids = list(set(metadata_sample_ids).difference(obs_sample_ids))
    # we first check if sample_IDs have been appended
    if appended_sample_ids:
        error_msg = "Appending data only in the metadata is not possible. "
        error_msg += "Please concatenate datasets first"
        raise ValueError(error_msg)

    removed_sample_ids = list(set(obs_sample_ids).difference(metadata_sample_ids))
    if removed_sample_ids:
        idxs = adata[adata.obs["sample_ID"].isin(metadata_sample_ids),:].obs.index.to_numpy()
        adata._inplace_subset_obs(idxs)
        adata.obs = _sanitize_categoricals(adata.obs)
    _reset_hash(adata, "adata_sample_ids")

    return

def _sync_sample_ids_from_obs(adata: AnnData) -> None:
    """\
    This function synchronizes the dataset by selecting the present sample_IDs
    in `adata.obs["sample_ID"]` and subsetting the metadata stored in `adata.uns["metadata"]`
    accordingly.

    Parameters
    ----------
    adata: AnnData
        the current dataset with an uns dict
    

    Examples
    --------
    >>> dataset = fp.create_dataset(...)
    >>> dataset = dataset[dataset.obs["sample_ID"].isin(["1", "2", "3"]),:]
    >>> fp.sync.sample_ids_from_obs_to_metadata(dataset)


    """


    print("\t... synchronizing metadata object to contain sample_IDs of the dataset")
    metadata_frame: pd.DataFrame = adata.uns["metadata"].dataframe.copy()
    obs = adata.obs

    metadata_sample_ids = metadata_frame["sample_ID"].tolist()
    obs_sample_ids = obs["sample_ID"].tolist()

    appended_sample_ids = list(set(obs_sample_ids).difference(metadata_sample_ids))
    # we first check if sample_IDs have been appended
    if appended_sample_ids:
        # we first have to grab all columns that
        # should be transferred to the metadata.
        # As this is not meant to synchronize columns,
        # we simply pull all columns that are currently
        # in the metadata and in adata.obs

        metadata_columns = adata.uns["metadata"].dataframe.columns
        columns_to_transfer = [col for col in metadata_columns
                               if col in adata.obs.columns]
        obs = adata.obs[columns_to_transfer]
        metadata = obs.drop_duplicates()
        
        # we make sure that the drop duplicates resulted
        # in a dataframe that only contains one line per sample_ID
        assert metadata.shape[0] == len(metadata["sample_ID"].unique())

        metadata_to_append = metadata[metadata["sample_ID"].isin(appended_sample_ids)]

        # because its possible that there are columns in the
        # metadata that are not present in obs,
        # we used a concat and dont calculate the metadata
        # from the obs_frame 
        metadata_frame = pd.concat(
            [adata.uns["metadata"].dataframe, metadata_to_append],
            axis = 0
        )

    metadata_sample_ids = metadata_frame["sample_ID"].tolist()
    removed_sample_ids = list(set(metadata_sample_ids).difference(obs_sample_ids))
    if removed_sample_ids:
        metadata_frame = metadata_frame[metadata_frame["sample_ID"].isin(obs_sample_ids)]
        metadata_frame = metadata_frame.sort_values("sample_ID", ascending = True)
        
    adata.uns["metadata"] = Metadata(metadata = metadata_frame)

    _reset_hash(adata, "metadata_sample_ids")
    return

def _sanitize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            df[column] = df[column].cat.remove_unused_categories()
    return df

from anndata import AnnData
import pandas as pd
import numpy as np

from typing import Optional

from ..dataset._supplements import Metadata

def _uncategorize_obs(obs: pd.DataFrame) -> pd.DataFrame:
    for column in obs.columns:
        print
        if isinstance(obs[column].dtype, pd.CategoricalDtype):
            obs[column] = obs[column].astype(obs[column].dtype._categories.dtype)
    return obs

def _categorize_obs(obs: pd.DataFrame) -> pd.DataFrame:
    return obs.astype("category")

def _get_metadata_from_dataset(adata: AnnData,
                               copy: bool = True) -> tuple[Metadata, pd.DataFrame]:
    
    metadata: Metadata = adata.uns["metadata"]
    adata_obs = adata.obs.copy() if copy else adata.obs
    
    return metadata, adata_obs

def sync_metadata_from_obs(adata: AnnData,
                           copy: bool = False) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    metadata, adata_obs = _get_metadata_from_dataset(adata,
                                                     copy = True)
    user_defined_metadata = metadata.dataframe.columns
    condensed_obs = adata_obs.drop_duplicates(subset = user_defined_metadata)
    condensed_obs = condensed_obs.sort_values("sample_ID", ascending = True)
    condensed_obs = condensed_obs.reset_index(drop = True)
    adata.uns["metadata"] = Metadata(metadata = condensed_obs)

    return adata if copy else None

def sync_metadata_from_uns(adata: AnnData,
                           copy: bool = False) -> Optional[AnnData]:
    
    adata = adata.copy() if copy else adata
    metadata, adata_obs = _get_metadata_from_dataset(adata,
                                                     copy = True)
    user_defined_metadata = metadata.dataframe.columns
    for col in user_defined_metadata:
        if col not in adata_obs.columns:
            adata_obs[col] = np.zeros(adata_obs.shape[0])

    adata_obs = adata_obs[user_defined_metadata]
    adata_obs = _uncategorize_obs(obs = adata_obs)

    for sample_ID in metadata.dataframe["sample_ID"]:
        adata_obs.loc[adata_obs["sample_ID"] == sample_ID, user_defined_metadata] = \
            metadata.dataframe.loc[metadata.dataframe["sample_ID"] == sample_ID, user_defined_metadata].values.flatten()

    adata_obs = _categorize_obs(obs = adata_obs)
    adata.obs = adata_obs

    return adata if copy else None
import pandas as pd
from ..dataset.supplements import Metadata
from anndata import AnnData
from typing import Optional

def synchronize_samples(adata: AnnData,
                        recalculate: bool = False) -> None:
    """
    Samples are synchronized so that only samples are kept that
    are the current dataset sample_IDs.

    First, the metadata are subset for the unique sampleIDs in
    adata.obs["sample_ID"]



    Args:
        adata (AnnData): _description_
        current_sample_IDs (pd.Series): _description_
    """
    #TODO: sanitize categoricals!

    current_obs_sample_IDs = adata.obs["sample_ID"].unique()

    _synchronize_metadata(adata,
                          current_obs_sample_IDs)

    for uns_frame in ["mfi_sample_ID_compensated",
                      "mfi_sample_ID_transformed",
                      "fop_sample_ID_compensated",
                      "fop_sample_ID_transformed"]:
        if uns_frame in adata.uns:
            if recalculate:
                _placeholder()
            _synchronize_uns_frame(adata,
                                   identifier = uns_frame,
                                   first_level_subset = current_obs_sample_IDs)


def _placeholder(): pass


def _synchronize_uns_frame(adata: AnnData,
                           identifier: str,
                           first_level_subset: Optional[pd.Series] = None,
                           second_level_subset: Optional[list[str]] = None) -> None:
    df: pd.DataFrame = adata.uns[identifier]
    if first_level_subset is None:
        first_level_subset = slice(None)
    if second_level_subset is None:
        second_level_subset = slice(None)
    adata.uns[identifier] = df.T.loc[(first_level_subset, second_level_subset),:].T

def _synchronize_metadata(adata: AnnData,
                          current_obs_sample_IDs: pd.Series) -> None:
    metadata: Metadata = adata.uns["metadata"]
    metadata.subset("sample_ID", current_obs_sample_IDs)
    return
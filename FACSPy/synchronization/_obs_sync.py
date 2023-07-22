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

    print("... Synchronizing metadata")
    _synchronize_metadata(adata,
                          current_obs_sample_IDs)

    for uns_frame in ["mfi_sample_ID_compensated",
                      "mfi_sample_ID_transformed",
                      "fop_sample_ID_compensated",
                      "fop_sample_ID_transformed",
                      "gate_frequencies"]:
        if uns_frame in adata.uns:
            print(f"... Synchronizing dataframe {uns_frame}")
            if recalculate:
                _placeholder()
            _synchronize_uns_frame(adata,
                                   identifier = uns_frame,
                                   sample_IDs = current_obs_sample_IDs)


def _placeholder(): pass

def extract_data_group_from_identifier(identifier):
    data_metric = identifier.split("_")[0]
    if "compensated" in identifier:
        return identifier.split(f"{data_metric}_")[1].split("_compensated")[0]
    if "transformed" in identifier:
        return identifier.split(f"{data_metric}_")[1].split("_transformed")[0]

def _synchronize_uns_frame(adata: AnnData,
                           identifier: str,
                           sample_IDs: list[str]) -> None:
    data_group = extract_data_group_from_identifier(identifier)
    if data_group is None or "sample_ID" in data_group:
        df: pd.DataFrame = adata.uns[identifier]
        adata.uns[identifier] = df.loc[df.index.get_level_values("sample_ID").isin(sample_IDs),:]

def _synchronize_metadata(adata: AnnData,
                          current_obs_sample_IDs: pd.Series) -> None:
    metadata: Metadata = adata.uns["metadata"]
    metadata.subset("sample_ID", current_obs_sample_IDs)
    return
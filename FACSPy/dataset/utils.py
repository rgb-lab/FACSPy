import pandas as pd
import numpy as np
from anndata import AnnData
from .supplements import Metadata, Panel
from typing import Union, Literal, Optional
from KDEpy import FFTKDE

def create_supplements(input_directory: str,
                       save_to_disk: bool = True) -> Optional[tuple[pd.DataFrame]]:
    metadata = Metadata(input_directory, from_fcs = True)
    panel = Panel(input_directory, from_fcs = True)
    return


def match_cell_numbers(adata: AnnData) -> AnnData:
    return adata

def create_sample_subset_with_controls(adata: AnnData,
                                       sample: str,
                                       corresponding_controls: dict,
                                       match_cell_number: bool) -> AnnData:
    controls: list[str] = corresponding_controls[sample]
    sample_list = [sample] + controls
    if match_cell_number:
        return match_cell_numbers(adata[adata.obs["file_name"].isin(sample_list)])
    return adata[adata.obs["file_name"].isin(sample_list)]

def transform_data_array(compensated_data: np.ndarray,
                         cofactors: Union[np.ndarray, int, float]) -> np.ndarray:
    return np.arcsinh(np.divide(compensated_data, cofactors))

def get_histogram_curve(data_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, x = np.histogram(data_array, bins = 100)
    _, curve = FFTKDE(kernel = "gaussian",
                        bw = "silverman"
                        ).fit(data_array).evaluate(100)
    return x, curve

def get_control_samples(dataframe: pd.DataFrame,
                        by: Literal["sample_ID", "file_name"]) -> list[str]:
    return dataframe.loc[dataframe["staining"] != "stained", by].to_list()

def get_stained_samples(dataframe: pd.DataFrame,
                        by: Literal["sample_ID", "file_name"]) -> list[str]:
    return dataframe.loc[dataframe["staining"] == "stained", by].to_list()


def reindex_metadata(metadata: pd.DataFrame,
                     indices: list[str]) -> pd.DataFrame:
    return metadata.set_index(indices) if indices else metadata

def find_name_of_control_sample_by_metadata(sample,
                                            metadata_to_compare: pd.DataFrame,
                                            indexed_frame: pd.DataFrame,
                                            by = Literal["sample_ID", "file_name"]) -> list[str]:
    matching_metadata = indexed_frame.loc[tuple(metadata_to_compare.values[0])]
    return matching_metadata.loc[matching_metadata[by] != sample, by].to_list()

def find_corresponding_control_samples(adata: AnnData,
                                       by: Literal["file_name", "sample_ID"]) -> tuple[list[str], dict[str, str]]:
    corresponding_controls = {}
    metadata: Metadata = adata.uns["metadata"]
    metadata_frame = metadata.to_df()
    metadata_factors = metadata.get_factors()
    indexed_metadata = reindex_metadata(metadata_frame,
                                        metadata_factors)
    
    stained_samples = get_stained_samples(metadata_frame,
                                          by = by)
    control_samples = get_control_samples(metadata_frame,
                                          by = by)
    for sample in stained_samples:
        if not metadata_factors:
            corresponding_controls[sample] = control_samples
            continue
        sample_metadata = metadata_frame.loc[metadata_frame[by] == sample, metadata.get_factors()]
        matching_control_samples = find_name_of_control_sample_by_metadata(sample,
                                                                           sample_metadata,
                                                                           indexed_metadata,
                                                                           by = by)
        corresponding_controls[sample] = matching_control_samples or control_samples
    return stained_samples, corresponding_controls
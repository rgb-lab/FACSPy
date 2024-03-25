import warnings

import os
import pandas as pd
import numpy as np
from anndata import AnnData
from KDEpy import FFTKDE
from flowutils import transforms

from typing import Union, Literal, Optional

from ._supplements import Metadata, CofactorTable, Panel

def _replace_missing_cofactors(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ 
    Missing cofactors can indicate Scatter-Channels and Time Channels
    or not-measured channels. In any case, cofactor is set to 1 for now.
    """
    dataframe[["cofactors"]] = dataframe[["cofactors"]].fillna(1)
    return dataframe

def _merge_cofactors_into_dataset_var(adata: AnnData,
                                     cofactor_table: CofactorTable) -> pd.DataFrame:
    if "cofactors" in adata.var.columns:
        adata.var = adata.var.drop("cofactors", axis = 1)
    adata_var = pd.merge(adata.var,
                         cofactor_table.dataframe,
                         left_on = "pns",
                         right_on = "fcs_colname",
                         how = "left")
    adata_var["cofactors"] = adata_var["cofactors"].astype("float") ## could crash, not tested, your fault if shitty
    adata_var.index = adata_var["pns"].to_list()
    adata_var = adata_var.drop("fcs_colname", axis = 1)
    adata_var["cofactors"] = adata_var["cofactors"].astype(np.float32)
    return adata_var 

def match_cell_numbers(adata: AnnData) -> AnnData:
    min_cell_number = adata.obs["file_name"].value_counts().min()
    return adata[adata.obs.groupby("file_name", observed = True).sample(n=min_cell_number).index,:]

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
    return asinh_transform(compensated_data, cofactors)

def _get_histogram_curve(data_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, x = np.histogram(data_array, bins = 100)
    try:
        _, curve = FFTKDE(kernel = "gaussian",
                          bw = "ISJ"
                          ).fit(data_array).evaluate(100)
    except ValueError: # signals a finite support error
        warnings.warn("Gaussian Kernel led to a value error, switching to Epanechnikov Kernel")
        _, curve = FFTKDE(kernel = "epa",
                          bw = "ISJ"
                          ).fit(data_array).evaluate(100)

    return x, curve

def get_control_samples(dataframe: pd.DataFrame,
                        by: Literal["sample_ID", "file_name"]) -> list[str]:
    return dataframe.loc[dataframe["staining"] != "stained", by].tolist()

def get_stained_samples(dataframe: pd.DataFrame,
                        by: Literal["sample_ID", "file_name"]) -> list[str]:
    return dataframe.loc[dataframe["staining"] == "stained", by].tolist()

def reindex_metadata(metadata: pd.DataFrame,
                     indices: list[str]) -> pd.DataFrame:
    return metadata.set_index(indices).sort_index()

def find_name_of_control_sample_by_metadata(sample,
                                            metadata_to_compare: pd.DataFrame,
                                            indexed_frame: pd.DataFrame,
                                            by = Literal["sample_ID", "file_name"]) -> list[str]:
    matching_metadata = indexed_frame.loc[tuple(metadata_to_compare.values[0])]
    return matching_metadata.loc[matching_metadata[by] != sample, by].tolist()

def asinh_transform(data: np.ndarray,
                    cofactors: Union[np.ndarray, int, float]) -> np.ndarray:
    """\
    Transforms the data according to the asinh function. First, the data
    are divided by the cofactors. The np.arcsinh function is then applied.

    Parameters
    ----------

    data
        The data to be transformed

    cofactors
        the cofactors for each channel

    Returns
    -------
        The transformed data array

    """
    return np.arcsinh(np.divide(data, cofactors))

def logicle_transform(data: np.ndarray,
                      m: Union[float, int],
                      t: Union[float, int],
                      w: Union[float, int],
                      a: Union[float, int],
                      channel_indices: np.ndarray) -> np.ndarray:
    """\
    Transforms the data according to the log transform.

    Parameters
    ----------

    data
        The data to be transformed

    channel_indices
        the indices of the channels to be transformed

    t
        parameter for the top of the linear scale (e.g. 262144)

    m
        parameter for desired number of decades
        
    w
        parameter for the approximate number of decades in the linear region

    a
        parameter for the additional number of negative decades

    Returns
    -------
        The transformed data array

    """
    return transforms.logicle(data,
                              channel_indices = channel_indices,
                              m = m,
                              t = t,
                              w = w,
                              a = a)

def hyperlog_transform(data: np.ndarray,
                       m: Union[float, int],
                       t: Union[float, int],
                       w: Union[float, int],
                       a: Union[float, int],
                       channel_indices: np.ndarray) -> np.ndarray:
    """\
    Transforms the data according to the log transform.

    Parameters
    ----------

    data
        The data to be transformed

    channel_indices
        the indices of the channels to be transformed

    t
        parameter for the top of the linear scale (e.g. 262144)

    m
        parameter for desired number of decades
        
    w
        parameter for the approximate number of decades in the linear region

    a
        parameter for the additional number of negative decades

    Returns
    -------
        The transformed data array

    """
    return transforms.hyperlog(data,
                               channel_indices = channel_indices,
                               m = m,
                               t = t,
                               w = w,
                               a = a)

def log_transform(data: np.ndarray,
                  m: Union[float, int],
                  t: Union[float, int],
                  channel_indices: np.ndarray) -> np.ndarray:
    """\
    Transforms the data according to the log transform.

    Parameters
    ----------

    data
        The data to be transformed

    channel_indices
        the indices of the channels to be transformed

    t
        parameter for the top of the linear scale (e.g. 262144)

    m
        parameter for desired number of decades

    Returns
    -------
        The transformed data array

    """
    return transforms.log(data,
                          channel_indices = channel_indices,
                          m = m,
                          t = t)

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
        if not metadata_factors or not control_samples:
            corresponding_controls[sample] = control_samples
            continue
        sample_metadata = metadata_frame.loc[metadata_frame[by] == sample, metadata.get_factors()]
        matching_control_samples = find_name_of_control_sample_by_metadata(sample,
                                                                           sample_metadata,
                                                                           indexed_metadata,
                                                                           by = by)
        corresponding_controls[sample] = matching_control_samples or control_samples
    return stained_samples, corresponding_controls

def _gather_fcs_files(input_directory: str) -> list[str]:
    return [file for file in os.listdir(input_directory)
            if file.endswith(".fcs")]

def create_empty_metadata(input_directory: Optional[str] = None,
                          as_frame: bool = False,
                          save: bool = True,
                          overwrite: bool = False) -> Union[pd.DataFrame, Metadata]:
    """\
    Creates a Metadata object from all .fcs files within a directory.
    The table will contain a sample_ID and the file_names.

    Parameters
    ----------

    input_directory
        The directory to be used. If no input_directory is specified,
        the current working directory is used.
    as_frame
        Whether to return the metadata as a pandas dataframe
    save
        Whether to save the metadata in the input_directory. Defaults to True.
        Will create a file called `metadata.csv`.
    overwrite
        Whether to overwrite the file `metadata.csv`, if the file already exists.
        Defaults to False

    Returns
    -------
    If `as_frame==True` a :class:`~pandas.DataFrame`, else a :class:`~FACSPy.dataset._supplements.Metadata` object

    Examples
    --------

    >>> import FACSPy as fp
    >>> metadata = fp.create_empty_metadata() # will read all file names with `.fcs` in the current working directory
    >>> fp.create_dataset(
    ...     metadata = metadata,
    ...     ...
    ... )

    """
    
    if input_directory is None:
        input_directory = os.getcwd()
    if not os.path.exists(input_directory):
        raise ValueError("Input directory not found")
    fcs_files = _gather_fcs_files(input_directory)
    df = pd.DataFrame(data = {
        "sample_ID": list(range(1,len(fcs_files)+1)),
        "file_name": fcs_files,
    })
    if save:
        print(f"... saving dataframe to {input_directory}.")
        if os.path.isfile(os.path.join(input_directory, "metadata.csv")) and not overwrite:
            raise FileExistsError("File already exists. Please set `overwrite` to True or `save` to False")
        df.to_csv(os.path.join(input_directory, "metadata.csv"), index = False)
    if as_frame:
        return df
    return Metadata(metadata = df)

def create_panel_from_fcs(input_directory: Optional[str],
                          as_frame: bool = False,
                          save: bool = True,
                          overwrite: bool = False) -> Union[pd.DataFrame, Panel]:
    """\
    Creates a Panel object from all .fcs files within a directory.
    The table will contain the channel names and the antigens stored in the FCS file.
    Note that all .fcs files have to contain the same panel, as only the first
    .fcs file is read and analyzed.

    Parameters
    ----------

    input_directory
        The directory to be used. If no input_directory is specified,
        the current working directory is used.
    as_frame
        Whether to return the panel as a pandas dataframe
    save
        Whether to save the panel in the input_directory. Defaults to True.
        Will create a file called `panel.csv`.
    overwrite
        Whether to overwrite the file `panel.csv`, if the file already exists.
        Defaults to False

    Returns
    -------
    If `as_frame==True` a :class:`~pandas.DataFrame`, else a :class:`~FACSPy.dataset._supplements.Panel` object

    Examples
    --------

    >>> import FACSPy as fp
    >>> metadata = fp.create_empty_metadata() # will read all file names with `.fcs` in the current working directory
    >>> fp.create_dataset(
    ...     metadata = metadata,
    ...     ...
    ... )

    """

    if input_directory is None:
        input_directory = os.getcwd()
    if not os.path.exists(input_directory):
        raise ValueError("Input Directory not found")
    fcs_files = _gather_fcs_files(input_directory)
    from ._sample import FCSFile
    print(f"... extracting panel information from the first FCS file {fcs_files[0]}")
    file = FCSFile(input_directory = input_directory,
                   file_name = fcs_files[0])
    channels = file.channels
    df = pd.DataFrame(data = {
        "fcs_colname": channels.index.tolist(),
        "antigens": channels["pns"].tolist()
    })
    if save:
        print(f"... saving panel to {input_directory}")
        if os.path.isfile(os.path.join(input_directory, "panel.csv")) and not overwrite:
            raise FileExistsError("File already exists. Please set `overwrite` to True or `save` to False")
        df.to_csv(os.path.join(input_directory, "panel.csv"),
                  index = False)
    if as_frame:
        return channels
    return Panel(panel = df)

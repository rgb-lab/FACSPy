import pytest
import os
import pandas as pd
import numpy as np

from anndata import AnnData
import anndata as ad
import FACSPy as fp
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.synchronization._utils import _sync_uns_frames
from FACSPy.synchronization._sample_sync import (_sync_sample_ids_from_obs,
                                                 _sync_sample_ids_from_metadata,
                                                 _sync_columns_from_obs,
                                                 _sync_columns_from_metadata)
from FACSPy.synchronization._synchronize import _dataset_has_been_modified

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(os.path.join(INPUT_DIRECTORY, "panel.txt"))
    metadata = Metadata(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"))
    workspace = FlowJoWorkspace(os.path.join(INPUT_DIRECTORY, "test_suite.wsp"))
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture
def mock_dataset():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    return fp.create_dataset(input_directory = input_directory,
                             panel = panel,
                             metadata = metadata,
                             workspace = workspace,
                             subsample_fcs_to = 100)

@pytest.fixture
def mock_dataset_analyzed():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace,
                              subsample_fcs_to = 300)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    fp.tl.mfi(adata)
    fp.tl.mds_samplewise(adata)
    return adata

def test_sync_sample_ids_from_obs_removed_samples(mock_dataset: AnnData):
    # we subset from adata
    adata = mock_dataset[mock_dataset.obs["sample_ID"].isin(["1", "2", "3"])].copy()

    _sync_sample_ids_from_obs(adata)
    remaining_sample_IDs = adata.uns["metadata"].dataframe["sample_ID"].tolist()

    assert len(remaining_sample_IDs) == 3
    assert all(sid in remaining_sample_IDs for sid in ["1", "2", "3"])
    assert not any(sid in remaining_sample_IDs for sid in ["4", "5", "6"])

def test_sync_sample_ids_from_obs_appended_samples(mock_dataset: AnnData):
    """
    this test covers the simplest case where sample_IDs are appended to
    the dataset. These sample_IDs are then transferred to the metadata
    """
    adata1 = mock_dataset
    adata2 = mock_dataset.copy()
    adata2.obs["sample_ID"] = adata2.obs["sample_ID"].map({"1": "7",
                                                           "2": "8",
                                                           "3": "9",
                                                           "4": "10",
                                                           "5": "11",
                                                           "6": "12"})
    del adata2.uns

    full_data = ad.concat(adatas = [adata1, adata2])
    full_data.obs_names_make_unique()
    full_data.uns = adata1.uns

    _sync_sample_ids_from_obs(full_data)
    metadata: pd.DataFrame = full_data.uns["metadata"].dataframe
    remaining_sample_IDs = metadata["sample_ID"].tolist()
    assert len(remaining_sample_IDs) == 12
    assert all(sid in remaining_sample_IDs for sid in [str(i) for i in range(1,12,1)])

    compare_cols = [col for col in metadata.columns if not "sample_ID" in col]
    assert np.array(metadata.loc[metadata["sample_ID"] == "1",compare_cols].values == metadata.loc[metadata["sample_ID"] == "7",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "2",compare_cols].values == metadata.loc[metadata["sample_ID"] == "8",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "3",compare_cols].values == metadata.loc[metadata["sample_ID"] == "9",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "4",compare_cols].values == metadata.loc[metadata["sample_ID"] == "10",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "5",compare_cols].values == metadata.loc[metadata["sample_ID"] == "11",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "6",compare_cols].values == metadata.loc[metadata["sample_ID"] == "12",compare_cols].values).all()

def test_sync_sample_ids_from_obs_appended_and_removed_samples(mock_dataset: AnnData):
    """
    this test covers the simplest case where sample_IDs are appended to
    the dataset. These sample_IDs are then transferred to the metadata
    """
    adata1 = mock_dataset
    adata2 = mock_dataset.copy()
    adata2.obs["sample_ID"] = adata2.obs["sample_ID"].map({"1": "7",
                                                           "2": "8",
                                                           "3": "9",
                                                           "4": "10",
                                                           "5": "11",
                                                           "6": "12"})
    del adata2.uns

    full_data = ad.concat(adatas = [adata1, adata2])
    full_data.obs_names_make_unique()
    full_data.uns = adata1.uns

    full_data = full_data[full_data.obs["sample_ID"] != "3",:].copy()
    print(full_data.obs["sample_ID"].unique())

    _sync_sample_ids_from_obs(full_data)
    metadata: pd.DataFrame = full_data.uns["metadata"].dataframe
    remaining_sample_IDs = metadata["sample_ID"].tolist()
    assert len(remaining_sample_IDs) == 11
    assert all(sid in remaining_sample_IDs
               for sid in [str(i) for i in range(1,12,1)] if sid != "3")

    compare_cols = [col for col in metadata.columns if not "sample_ID" in col]
    assert np.array(metadata.loc[metadata["sample_ID"] == "1",compare_cols].values == metadata.loc[metadata["sample_ID"] == "7",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "2",compare_cols].values == metadata.loc[metadata["sample_ID"] == "8",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "4",compare_cols].values == metadata.loc[metadata["sample_ID"] == "10",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "5",compare_cols].values == metadata.loc[metadata["sample_ID"] == "11",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "6",compare_cols].values == metadata.loc[metadata["sample_ID"] == "12",compare_cols].values).all()

def test_sync_sample_ids_from_obs_appended_samples_2(mock_dataset: AnnData):
    """
    this test covers the simplest case where sample_IDs are appended to
    the dataset. These sample_IDs are then transferred to the metadata.
    This test in addition covers the case where there is an additional column
    in the obs data. This column should not be transferred if only
    sample IDs are transferred.
    """
    adata1 = mock_dataset
    adata2 = mock_dataset.copy()
    adata2.obs["sample_ID"] = adata2.obs["sample_ID"].map({"1": "7",
                                                           "2": "8",
                                                           "3": "9",
                                                           "4": "10",
                                                           "5": "11",
                                                           "6": "12"})

    del adata2.uns
    full_data = ad.concat(adatas = [adata1, adata2])
    full_data.obs_names_make_unique()
    full_data.uns = adata1.uns
    full_data.obs["random_data"] = full_data.obs["sample_ID"].map({"1": "what",
                                                                   "2": "are",
                                                                   "3": "these",
                                                                   "4": "data",
                                                                   "5": "anyway",
                                                                   "6": "?",
                                                                   "7": "what",
                                                                   "8": "are",
                                                                   "9": "these",
                                                                   "10": "data",
                                                                   "11": "anyway",
                                                                   "12": "?"})
    _sync_sample_ids_from_obs(full_data)
    metadata: pd.DataFrame = full_data.uns["metadata"].dataframe
    remaining_sample_IDs = metadata["sample_ID"].tolist()

    assert "random_data" not in metadata.columns

    assert len(remaining_sample_IDs) == 12
    assert all(sid in remaining_sample_IDs for sid in [str(i) for i in range(1,12,1)])

    compare_cols = [col for col in metadata.columns if not "sample_ID" in col]
    assert np.array(metadata.loc[metadata["sample_ID"] == "1",compare_cols].values == metadata.loc[metadata["sample_ID"] == "7",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "2",compare_cols].values == metadata.loc[metadata["sample_ID"] == "8",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "3",compare_cols].values == metadata.loc[metadata["sample_ID"] == "9",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "4",compare_cols].values == metadata.loc[metadata["sample_ID"] == "10",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "5",compare_cols].values == metadata.loc[metadata["sample_ID"] == "11",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "6",compare_cols].values == metadata.loc[metadata["sample_ID"] == "12",compare_cols].values).all()

def test_sync_sample_ids_from_obs_appended_samples_3(mock_dataset: AnnData):
    """
    this test covers the simplest case where sample_IDs are appended to
    the dataset. These sample_IDs are then transferred to the metadata.
    This test in addition covers the case where there is an additional column
    in the metadata. This column should stay there but filled with NaN.
    """
    adata1 = mock_dataset
    adata2 = mock_dataset.copy()
    adata1.obs["random_data"] = adata1.obs["sample_ID"].map({"1": "what",
                                                             "2": "are",
                                                             "3": "these",
                                                             "4": "data",
                                                             "5": "anyway",
                                                             "6": "?"})
    adata1.uns["metadata"].dataframe["random_data"] = adata1.uns["metadata"].dataframe["sample_ID"].map({"1": "what",
                                                                                                         "2": "are",
                                                                                                         "3": "these",
                                                                                                         "4": "data",
                                                                                                         "5": "anyway",
                                                                                                         "6": "?"})
    adata2.obs["sample_ID"] = adata2.obs["sample_ID"].map({"1": "7",
                                                           "2": "8",
                                                           "3": "9",
                                                           "4": "10",
                                                           "5": "11",
                                                           "6": "12"})

    del adata2.uns
    full_data = ad.concat(adatas = [adata1, adata2])
    full_data.obs_names_make_unique()
    full_data.uns = adata1.uns
    _sync_sample_ids_from_obs(full_data)
    metadata: pd.DataFrame = full_data.uns["metadata"].dataframe
    remaining_sample_IDs = metadata["sample_ID"].tolist()

    assert "random_data" in metadata.columns
    assert metadata.loc[metadata["sample_ID"].isin(adata2.obs["sample_ID"].unique()),"random_data"].isna().all()

    assert len(remaining_sample_IDs) == 12
    assert all(sid in remaining_sample_IDs for sid in [str(i) for i in range(1,12,1)])

    compare_cols = [col for col in metadata.columns if not any(k in col for k in ["sample_ID", "random_data"])]
    assert np.array(metadata.loc[metadata["sample_ID"] == "1",compare_cols].values == metadata.loc[metadata["sample_ID"] == "7",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "2",compare_cols].values == metadata.loc[metadata["sample_ID"] == "8",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "3",compare_cols].values == metadata.loc[metadata["sample_ID"] == "9",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "4",compare_cols].values == metadata.loc[metadata["sample_ID"] == "10",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "5",compare_cols].values == metadata.loc[metadata["sample_ID"] == "11",compare_cols].values).all()
    assert np.array(metadata.loc[metadata["sample_ID"] == "6",compare_cols].values == metadata.loc[metadata["sample_ID"] == "12",compare_cols].values).all()

def test_synchronize_obs_wo_recalc(mock_dataset_analyzed: AnnData):
    adata = mock_dataset_analyzed
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df.columns
    adata = adata[adata.obs["condition1"].isin(["x", "y"]),:].copy()
    assert _dataset_has_been_modified(adata)
    _sync_uns_frames(adata,
                     recalculate = False)
    new_sample_IDs = adata.obs["sample_ID"].unique().tolist()
    df_after_sync: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "1" in df_after_sync.index
    assert "2" in df_after_sync.index
    assert "3" in df_after_sync.index
    assert "4" in df_after_sync.index
    assert "5" not in df_after_sync.index
    assert "6" not in df_after_sync.index
    np.testing.assert_array_equal(
        df.loc[df.index.get_level_values("sample_ID").isin(new_sample_IDs), ["MDS1", "MDS2", "MDS3"]].values,
        df_after_sync.loc[df_after_sync.index.get_level_values("sample_ID").isin(new_sample_IDs), ["MDS1", "MDS2", "MDS3"]].values
    )

def test_synchronize_obs_w_recalc(mock_dataset_analyzed: AnnData):
    adata = mock_dataset_analyzed
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert all(k in df.columns for k in ["MDS1", "MDS2", "MDS3"])

    adata = adata[adata.obs["condition1"].isin(["x", "y"]),:].copy()
    assert _dataset_has_been_modified(adata)
    _sync_uns_frames(adata,
                     recalculate = True)
    new_sample_IDs = adata.obs["sample_ID"].unique().tolist()
    df_after_sync: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert all(k in df_after_sync.columns for k in ["MDS1", "MDS2", "MDS3"])
    assert all(k in df_after_sync.index for k in ["1", "2", "3", "4"])
    assert not any(k in df_after_sync.index for k in ["5", "6"])
    # because for the test dataset the last row is NaN, np.all would return False
    assert np.any(
        np.not_equal(
            df.loc[df.index.get_level_values("sample_ID").isin(new_sample_IDs), ["MDS1", "MDS2", "MDS3"]].values.flatten(),
            df_after_sync.loc[df_after_sync.index.get_level_values("sample_ID").isin(new_sample_IDs), ["MDS1", "MDS2", "MDS3"]].values.flatten()
        )
    )

def test_sync_metadata_from_obs(mock_dataset: AnnData):
    metadata = mock_dataset.uns["metadata"].dataframe
    metadata = metadata[metadata["sample_ID"].isin(["1", "2", "3"])]
    mock_dataset.uns["metadata"] = Metadata(metadata = metadata)
    assert mock_dataset.uns["metadata"].dataframe.shape[0] == 3

    _sync_sample_ids_from_metadata(mock_dataset)

    assert all(k in mock_dataset.obs["sample_ID"].unique().tolist() for k in ["1", "2", "3"])
    assert not any(k in mock_dataset.obs["sample_ID"].unique().tolist() for k in ["5", "6", "4"])

def test_sync_metadata_from_obs_valueerror(mock_dataset: AnnData):
    metadata: pd.DataFrame = mock_dataset.uns["metadata"].dataframe
    metadata_copy = metadata.copy()
    metadata_copy["sample_ID"] = metadata_copy["sample_ID"].map({"1": "7",
                                                                 "2": "8",
                                                                 "3": "9",
                                                                 "4": "10",
                                                                 "5": "11",
                                                                 "6": "12"})
    
    metadata = pd.concat([metadata, metadata_copy], axis = 0)
    mock_dataset.uns["metadata"] = Metadata(metadata = metadata)
    assert mock_dataset.uns["metadata"].dataframe.shape[0] == 12
    with pytest.raises(ValueError):
        _sync_sample_ids_from_metadata(mock_dataset)

def test_column_sync_from_obs(mock_dataset: AnnData):
    random_data_map = {
        "1": "7",
        "2": "8",
        "3": "9",
        "4": "10",
        "5": "11",
        "6": "12"
    }
    mock_dataset.obs["random_data"] = mock_dataset.obs["sample_ID"].map(random_data_map)
    _sync_columns_from_obs(mock_dataset)
    metadata: pd.DataFrame = mock_dataset.uns["metadata"].dataframe
    assert "random_data" in metadata.columns
    metadata_dict = dict(zip(metadata["sample_ID"], metadata["random_data"]))
    assert metadata_dict == random_data_map

def test_column_sync_from_metadata(mock_dataset: AnnData):
    random_data_map = {
        "1": "7",
        "2": "8",
        "3": "9",
        "4": "10",
        "5": "11",
        "6": "12"
    }
    mock_dataset.uns["metadata"].dataframe["random_data"] = mock_dataset.uns["metadata"].dataframe["sample_ID"].map(random_data_map)
    original_idxs = mock_dataset.obs_names
    _sync_columns_from_metadata(mock_dataset)
    assert "random_data" in mock_dataset.obs.columns
    new_idxs = mock_dataset.obs_names
    assert all(original_idxs == new_idxs)

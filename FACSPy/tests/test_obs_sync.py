import pytest
from anndata import AnnData
import pandas as pd
import numpy as np

import FACSPy as fp
from FACSPy.synchronization._obs_sync import synchronize_samples
from FACSPy.synchronization._synchronize import _dataset_has_been_modified
from FACSPy.dataset._supplements import Panel, Metadata, CofactorTable
from FACSPy.dataset._workspaces import FlowJoWorkspace

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(input_directory = INPUT_DIRECTORY,
                  file_name = "panel.txt")
    metadata = Metadata(input_directory = INPUT_DIRECTORY,
                        file_name = "metadata_test_suite.csv")
    workspace = FlowJoWorkspace(input_directory = INPUT_DIRECTORY,
                                file_name = "test_suite.wsp")
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture
def mock_dataset():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    fp.tl.mfi(adata)
    fp.tl.mds_samplewise(adata)
    return adata

def test_synchronize_obs_wo_recalc(mock_dataset: AnnData):
    adata = mock_dataset
    original_sample_IDs = adata.obs["sample_ID"].unique().tolist()
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df.columns
    adata = adata[adata.obs["condition1"].isin(["x", "y"]),:].copy()
    assert _dataset_has_been_modified(adata)
    synchronize_samples(adata,
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

def test_synchronize_obs_w_recalc(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df.columns
    assert "MDS2" in df.columns
    assert "MDS3" in df.columns
    adata = adata[adata.obs["condition1"].isin(["x", "y"]),:].copy()
    assert _dataset_has_been_modified(adata)
    synchronize_samples(adata,
                        recalculate = True)
    new_sample_IDs = adata.obs["sample_ID"].unique().tolist()
    df_after_sync: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df_after_sync.columns
    assert "MDS2" in df_after_sync.columns
    assert "MDS3" in df_after_sync.columns
    assert "1" in df_after_sync.index
    assert "2" in df_after_sync.index
    assert "3" in df_after_sync.index
    assert "4" in df_after_sync.index
    assert "5" not in df_after_sync.index
    assert "6" not in df_after_sync.index
    # because for the test dataset the last row is zeros, np.all would return False
    assert np.any(
        np.not_equal(
            df.loc[df.index.get_level_values("sample_ID").isin(new_sample_IDs), ["MDS1", "MDS2", "MDS3"]].values.flatten(),
            df_after_sync.loc[df_after_sync.index.get_level_values("sample_ID").isin(new_sample_IDs), ["MDS1", "MDS2", "MDS3"]].values.flatten()
        )
    )
            

def test_synchronize_obs_wo_recalc_metadata_object(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df.columns
    adata = adata[adata.obs["condition1"].isin(["x", "y"]),:].copy()
    synchronize_samples(adata,
                        recalculate = True)
    metadata = adata.uns["metadata"]
    assert metadata.dataframe["sample_ID"].tolist() == ["1", "2", "3", "4"]
    assert list(metadata.dataframe["sample_ID"].cat.categories) == ["1", "2", "3", "4"]
import pytest
import os

import numpy as np


from anndata import AnnData
import pandas as pd
import FACSPy as fp
from FACSPy.synchronization._utils import (_get_frame_metrics,
                                           _get_present_samplewise_dimreds,
                                           _get_samplewise_dimred_columns)
from FACSPy.synchronization._var_sync import synchronize_vars
from FACSPy.synchronization._synchronize import _dataset_has_been_modified
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace
from FACSPy._utils import remove_channel
from FACSPy.dataset._supplements import CofactorTable

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
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    fp.tl.mfi(adata,
              use_only_fluo = False)
    fp.tl.mds_samplewise(adata)
    return adata

def test_synchronize_vars_wo_recalc(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df.columns
    remove_channel(adata, channel = "CD15")
    assert _dataset_has_been_modified(adata)
    synchronize_vars(adata,
                     recalculate = False)
    df_after_sync: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df_after_sync.columns
    assert "CD15" not in df_after_sync.columns
    np.testing.assert_array_equal(df[["MDS1", "MDS2", "MDS3"]].values, df_after_sync[["MDS1", "MDS2", "MDS3"]].values)

def test_synchronize_vars_with_recalc(mock_dataset: AnnData):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df.columns
    remove_channel(adata, channel = "CD15")
    assert _dataset_has_been_modified(adata)
    synchronize_vars(adata,
                     recalculate = True)
    df_after_sync: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df_after_sync.columns
    assert "CD15" not in df_after_sync.columns
    # any needed because the last row of MDS coordinates is zero for all samples
    assert np.any(
        np.not_equal(df[["MDS1", "MDS2", "MDS3"]].values.flatten(),
                     df_after_sync[["MDS1", "MDS2", "MDS3"]].values.flatten()
        )
    )

def test_get_samplewise_dimred_columns(mock_dataset):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert "MDS1" in df.columns
    assert _get_samplewise_dimred_columns(df) == ["MDS1", "MDS2", "MDS3"]

def test_get_samplewise_dimreds(mock_dataset):
    adata = mock_dataset
    df: pd.DataFrame = adata.uns["mfi_sample_ID_compensated"]
    assert _get_present_samplewise_dimreds(df) == ["MDS"]

def test_get_frame_metrics():
    data_metric, data_origin, data_group = _get_frame_metrics("mfi_sample_ID_compensated")
    assert data_metric == "mfi"
    assert data_origin == "compensated"
    assert data_group == "sample_ID"

    data_metric, data_origin, data_group = _get_frame_metrics("fop_something_transformed")
    assert data_metric == "fop"
    assert data_origin == "transformed"
    assert data_group == "something"

    data_metric, data_origin, data_group = _get_frame_metrics("fop_some_long_string_that_a_user_gave_the_group_asinh")
    assert data_metric == "fop"
    assert data_origin == "asinh"
    assert data_group == "some_long_string_that_a_user_gave_the_group"

def test_panel_update(mock_dataset: AnnData):
    # cofactors need to be added manually here
    cofactors = CofactorTable(cofactors = pd.DataFrame(data = {"fcs_colname": mock_dataset.uns["panel"].dataframe["antigens"].to_list(),
                                                               "cofactors": [1 for _ in range(len(mock_dataset.uns["panel"].dataframe))]}))
    mock_dataset.uns["cofactors"] = cofactors
    fp.remove_channel(mock_dataset, "CD15")
    synchronize_vars(mock_dataset)
    assert "CD15" not in mock_dataset.uns["panel"].dataframe["antigens"].to_list()
    assert "CD15" not in mock_dataset.uns["cofactors"].dataframe["fcs_colname"].to_list()

    


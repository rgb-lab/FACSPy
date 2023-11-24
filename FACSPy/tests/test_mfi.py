import pytest
import os
from anndata import AnnData
import numpy as np
import pandas as pd

import FACSPy as fp
from FACSPy.tools._mfi import mfi
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace
from FACSPy._utils import find_gate_path_of_gate

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(os.path.join(INPUT_DIRECTORY, "panel.txt"))
    metadata = Metadata(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"))
    workspace = FlowJoWorkspace(os.path.join(INPUT_DIRECTORY, "test_suite.wsp"))
    return INPUT_DIRECTORY, panel, metadata, workspace


@pytest.fixture
def mock_dataset() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    return adata

def test_aliasing(mock_dataset: AnnData):
    fp.tl.mfi(mock_dataset)
    assert "mfi_sample_ID_compensated" in mock_dataset.uns

def test_copying(mock_dataset: AnnData):
    x = fp.tl.mfi(mock_dataset,
                  copy = True)
    assert isinstance(x, AnnData)

    x = fp.tl.mfi(mock_dataset,
                  copy = False)
    assert x is None

def test_mfi_function(mock_dataset: AnnData):
    mfi(mock_dataset,
        groupby = "sample_ID",
        layer = ["compensated", "transformed"])
    assert "mfi_sample_ID_compensated" in mock_dataset.uns
    assert "mfi_sample_ID_transformed" in mock_dataset.uns

def test_only_fluo(mock_dataset: AnnData):
    mfi(mock_dataset,
        use_only_fluo = False)
    mfi_frame = mock_dataset.uns["mfi_sample_ID_compensated"]
    assert "FSC-A" in mfi_frame.columns

def test_only_fluo(mock_dataset: AnnData):
    mfi(mock_dataset,
        use_only_fluo = True)
    mfi_frame = mock_dataset.uns["mfi_sample_ID_compensated"]
    assert "FSC-A" not in mfi_frame.columns

def test_dataframe_contents(mock_dataset: AnnData):
    mfi(mock_dataset,
        use_only_fluo = False)
    mfi_frame: pd.DataFrame = mock_dataset.uns["mfi_sample_ID_compensated"]
    assert list(mfi_frame.index.names) == ["sample_ID", "gate"]
    assert all([name in mfi_frame.columns for name in mock_dataset.var_names.tolist()])

def test_groupby_parameter(mock_dataset: AnnData):
    mfi(mock_dataset,
        groupby = "condition1",
        use_only_fluo = False)
    assert "mfi_condition1_compensated" in mock_dataset.uns
    mfi_frame: pd.DataFrame = mock_dataset.uns["mfi_condition1_compensated"]
    for index_entry in ["sample_ID", "condition1", "gate"]:
        assert index_entry in list(mfi_frame.index.names)
    
def test_error_for_wrong_metric(mock_dataset: AnnData):
    with pytest.raises(NotImplementedError):
        mfi(mock_dataset,
            method = "random_method")

def test_layer_setting(mock_dataset: AnnData):
    mock_dataset.layers["test_layer"] = mock_dataset.layers["compensated"].copy()
    mfi(mock_dataset,
        layer = "test_layer")
    assert "mfi_sample_ID_test_layer" in mock_dataset.uns

def test_correct_mfi_mean(mock_dataset: AnnData):
    specific_gate_adata = fp.subset_gate(mock_dataset, "live", copy = True)
    full_gate_path = find_gate_path_of_gate(mock_dataset, "live")
    df = specific_gate_adata.to_df(layer = "compensated")
    df["sample_ID"] = specific_gate_adata.obs["sample_ID"]
    mean_df = df.groupby("sample_ID").mean()
    mean_df["gate"] = full_gate_path
    mean_df = mean_df.set_index("gate", append = True)

    mfi(mock_dataset,
        use_only_fluo = False,
        method = "mean")
    
    facspy_calculated: pd.DataFrame = mock_dataset.uns["mfi_sample_ID_compensated"]
    gate_specific_data: pd.DataFrame = facspy_calculated.loc[facspy_calculated.index.get_level_values("gate") == full_gate_path,:]

    assert gate_specific_data.equals(mean_df)

def test_correct_mfi_median(mock_dataset: AnnData):
    specific_gate_adata = fp.subset_gate(mock_dataset, "live", copy = True)
    full_gate_path = find_gate_path_of_gate(mock_dataset, "live")
    df = specific_gate_adata.to_df(layer = "compensated")
    df["sample_ID"] = specific_gate_adata.obs["sample_ID"]
    mean_df = df.groupby("sample_ID").median()
    mean_df["gate"] = full_gate_path
    mean_df = mean_df.set_index("gate", append = True)

    mfi(mock_dataset,
        use_only_fluo = False,
        method = "median")
    
    facspy_calculated: pd.DataFrame = mock_dataset.uns["mfi_sample_ID_compensated"]
    gate_specific_data: pd.DataFrame = facspy_calculated.loc[facspy_calculated.index.get_level_values("gate") == full_gate_path,:]

    assert gate_specific_data.equals(mean_df)

def test_settings_save(mock_dataset: AnnData):
    mfi(mock_dataset,
        use_only_fluo = False,
        method = "median")
    assert "settings" in mock_dataset.uns
    settings = mock_dataset.uns["settings"]
    assert "_mfi_sample_ID_compensated" in settings
    assert settings["_mfi_sample_ID_compensated"]["groupby"] == "sample_ID"
    assert settings["_mfi_sample_ID_compensated"]["use_only_fluo"] == False
    assert settings["_mfi_sample_ID_compensated"]["method"] == "median"
    assert settings["_mfi_sample_ID_compensated"]["layer"] == "compensated"







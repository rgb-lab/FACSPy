import pytest
from anndata import AnnData
import pandas as pd

import FACSPy as fp
from FACSPy.tools._fop import fop
from FACSPy.dataset._supplements import Metadata, Panel, CofactorTable
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
def mock_dataset() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    cofactors = CofactorTable(input_directory = input_directory,
                              file_name = "cofactors_test_suite.txt")
    adata.uns["cofactors"] = cofactors
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    return adata

def test_aliasing(mock_dataset: AnnData):
    fp.tl.fop(mock_dataset)
    assert "fop_sample_ID_compensated" in mock_dataset.uns

def test_copying(mock_dataset: AnnData):
    x = fp.tl.fop(mock_dataset,
                  copy = True)
    assert isinstance(x, AnnData)

    x = fp.tl.fop(mock_dataset,
                  copy = False)
    assert x is None

def test_fop_function(mock_dataset: AnnData):
    fop(mock_dataset,
        groupby = "sample_ID")
    assert "fop_sample_ID_compensated" in mock_dataset.uns
    assert "fop_sample_ID_transformed" in mock_dataset.uns

def test_only_fluo(mock_dataset: AnnData):
    fop(mock_dataset,
        use_only_fluo = False)
    fop_frame = mock_dataset.uns["fop_sample_ID_compensated"]
    assert "FSC-A" in fop_frame.columns

def test_only_fluo(mock_dataset: AnnData):
    fop(mock_dataset,
        use_only_fluo = True)
    fop_frame: pd.DataFrame = mock_dataset.uns["fop_sample_ID_compensated"]
    assert "FSC-A" not in fop_frame.columns

def test_dataframe_contents(mock_dataset: AnnData):
    fop(mock_dataset,
        use_only_fluo = False)
    fop_frame: pd.DataFrame = mock_dataset.uns["fop_sample_ID_compensated"]
    assert list(fop_frame.index.names) == ["sample_ID", "gate"]
    assert all([name in fop_frame.columns for name in mock_dataset.var_names.tolist()])

def test_groupby_parameter(mock_dataset: AnnData):
    fop(mock_dataset,
        groupby = "condition1",
        use_only_fluo = False)
    assert "fop_condition1_compensated" in mock_dataset.uns
    fop_frame: pd.DataFrame = mock_dataset.uns["fop_condition1_compensated"]
    for index_entry in ["sample_ID", "condition1", "gate"]:
        assert index_entry in list(fop_frame.index.names)
    
def test_layer_setting(mock_dataset: AnnData):
    mock_dataset.layers["test_layer"] = mock_dataset.layers["compensated"].copy()
    fop(mock_dataset,
        layer = "test_layer")
    assert "fop_sample_ID_test_layer" in mock_dataset.uns

def test_cuttoff(mock_dataset: AnnData):
    fop(mock_dataset,
        cutoff = 1000)
    assert "fop_sample_ID_compensated" in mock_dataset.uns

def test_cuttoff(mock_dataset: AnnData):
    fop(mock_dataset,
        cutoff = [90+i for i in range(len(mock_dataset.var_names))])
    assert "fop_sample_ID_compensated" in mock_dataset.uns

def test_settings_save(mock_dataset: AnnData):
    fop(mock_dataset,
        use_only_fluo = False)
        
    assert "settings" in mock_dataset.uns
    settings = mock_dataset.uns["settings"]
    assert "_fop_sample_ID_compensated" in settings
    assert settings["_fop_sample_ID_compensated"]["groupby"] == "sample_ID"
    assert settings["_fop_sample_ID_compensated"]["use_only_fluo"] == False
    assert settings["_fop_sample_ID_compensated"]["cutoff"] == None
    assert all(settings["_fop_sample_ID_compensated"]["cofactors"] == mock_dataset.var["cofactors"].to_numpy())






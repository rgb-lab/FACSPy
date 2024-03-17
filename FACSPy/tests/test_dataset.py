import pytest
import os
import pandas as pd

from anndata import AnnData
from scipy.sparse import csr_matrix

import FACSPy as fp
from FACSPy.dataset import create_dataset
from FACSPy.dataset._dataset import DatasetAssembler

from FACSPy.exceptions._exceptions import InputDirectoryNotFoundError
from FACSPy.exceptions._supplements import SupplementDataTypeError

from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(os.path.join(INPUT_DIRECTORY, "panel.txt"))
    metadata = Metadata(os.path.join(INPUT_DIRECTORY, "metadata_test_suite.csv"))
    workspace = FlowJoWorkspace(os.path.join(INPUT_DIRECTORY, "test_suite.wsp"))
    return INPUT_DIRECTORY, panel, metadata, workspace

@pytest.fixture
def mock_dataset_assembler_object():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    return DatasetAssembler(input_directory = input_directory,
                            panel = panel,
                            metadata = metadata,
                            workspace = workspace)

@pytest.fixture
def mock_panel_correct():
    panel = pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "antigens": ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"],
        }
    )
    return Panel(panel = panel)

@pytest.fixture
def mock_metadata_correct():
    metadata = pd.DataFrame(
        {
            "sample_ID": list(range(2)),
            "file_name": ["some_file.fcs", "some_other_file.fcs"]
        }
    )
    return Metadata(metadata = metadata)

@pytest.fixture
def dummy_workspace():
    return FlowJoWorkspace(os.path.join(WSP_FILE_PATH, WSP_FILE_NAME))

@pytest.fixture
def mock_dataset():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    return fp.create_dataset(input_directory = input_directory,
                             panel = panel,
                             metadata = metadata,
                             workspace = workspace,
                             keep_raw = True)

@pytest.fixture
def mock_dataset_no_raw():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    return fp.create_dataset(input_directory = input_directory,
                             panel = panel,
                             metadata = metadata,
                             workspace = workspace,
                             keep_raw = False)
     

def test_create_dataset_invalid_input_directory(mock_metadata_correct,
                                                mock_panel_correct,
                                                dummy_workspace):
    with pytest.raises(InputDirectoryNotFoundError):
        _ = create_dataset(input_directory = "1&1231",
                           metadata = mock_metadata_correct,
                           panel = mock_panel_correct,
                           workspace = dummy_workspace) 

def test_create_dataset_wrong_input_workspace(mock_metadata_correct,
                                              mock_panel_correct):
    with pytest.raises(SupplementDataTypeError):
        _ = create_dataset(input_directory = os.getcwd(),
                           metadata = mock_metadata_correct,
                           panel = mock_panel_correct,
                           workspace = "1") 

def test_create_dataset_wrong_input_panel(mock_metadata_correct,
                                          dummy_workspace):
    with pytest.raises(SupplementDataTypeError):
        _ = create_dataset(input_directory = os.getcwd(),
                           metadata = mock_metadata_correct,
                           panel = "1",
                           workspace = dummy_workspace) 

def test_create_dataset_wrong_input_metadata(mock_panel_correct,
                                             dummy_workspace):
    with pytest.raises(SupplementDataTypeError):
        _ = create_dataset(input_directory = os.getcwd(),
                           metadata = "1",
                           panel = mock_panel_correct,
                           workspace = dummy_workspace)

def test_subsample_parameter():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = create_dataset(input_directory = input_directory,
                           panel = panel,
                           metadata = metadata,
                           workspace = workspace,
                           subsample_fcs_to = 20_000)
    assert adata.shape[0] == 120_000

def test_subsample_parameter_ValueError():
    input_directory, panel, metadata, workspace = create_supplement_objects()
    with pytest.raises(ValueError):
        adata = create_dataset(input_directory = input_directory,
                               panel = panel,
                               metadata = metadata,
                               workspace = workspace,
                               subsample_fcs_to = "something")

def test_dataset_size(mock_dataset: AnnData):
    assert mock_dataset.shape == (284937, 21)

def test_correct_gating_names(mock_dataset: AnnData):
    """gates have been read from flowjo"""
    assert "root/FSC_SSC" in mock_dataset.uns["gating_cols"]
    assert "root/FSC_SSC/FSC_singlets" in mock_dataset.uns["gating_cols"]
    assert "root/FSC_SSC/FSC_singlets/SSC_singlets" in mock_dataset.uns["gating_cols"]
    assert "root/FSC_SSC/FSC_singlets/SSC_singlets/live" in mock_dataset.uns["gating_cols"]
    assert "root/FSC_SSC/FSC_singlets/SSC_singlets/live/Neutrophils" in mock_dataset.uns["gating_cols"]
    assert "root/FSC_SSC/FSC_singlets/SSC_singlets/live/Neutrophils/file2_specific_gate" in mock_dataset.uns["gating_cols"]

def test_correct_gating_for_one_file(mock_dataset: AnnData): 
    fp.subset_gate(mock_dataset, "FSC_SSC")
    assert mock_dataset[mock_dataset.obs["sample_ID"] == "3",:].shape == (50619,21)
    assert len(mock_dataset.obs["sample_ID"].unique()) == 1

def test_dataset_layers(mock_dataset: AnnData):
    """tests to confirm the presence and identities of the anndata"""
    assert mock_dataset.X is None
    assert mock_dataset.layers
    assert "raw" in mock_dataset.layers
    assert "compensated" in mock_dataset.layers
    assert isinstance(mock_dataset.obsm["gating"], csr_matrix)

def test_dataset_layers_no_raw(mock_dataset_no_raw: AnnData):
    """tests to confirm the presence and identities of the anndata"""
    assert mock_dataset_no_raw.X is None
    assert mock_dataset_no_raw.layers
    assert "raw" not in mock_dataset_no_raw.layers
    assert "compensated" in mock_dataset_no_raw.layers
    assert isinstance(mock_dataset_no_raw.obsm["gating"], csr_matrix)

def test_dataset_obs(mock_dataset: AnnData):
    assert isinstance(mock_dataset.obs, pd.DataFrame)
    assert "sample_ID" in mock_dataset.obs.columns
    assert "file_name" in mock_dataset.obs.columns
    assert len(mock_dataset.obs) == 284937

def test_dataset_var(mock_dataset: AnnData):
    assert isinstance(mock_dataset.var, pd.DataFrame)
    assert "pns" in mock_dataset.var.columns
    assert "pnr" in mock_dataset.var.columns
    assert "pne" in mock_dataset.var.columns
    assert "type" in mock_dataset.var.columns
    assert "pnn" in mock_dataset.var.columns
    assert len(mock_dataset.var) == 21

def test_compensated_different_to_raw(mock_dataset: AnnData):
    import numpy as np
    np.not_equal(mock_dataset.layers["raw"].flatten(), mock_dataset.layers["compensated"].flatten())

def test_presence_of_uns_data(mock_dataset: AnnData):
    assert mock_dataset.uns["metadata"] is not None
    assert isinstance(mock_dataset.uns["metadata"], Metadata)
    assert mock_dataset.uns["panel"] is not None
    assert isinstance(mock_dataset.uns["panel"], Panel)
    assert mock_dataset.uns["workspace"] is not None
    assert isinstance(mock_dataset.uns["workspace"], dict)
    assert mock_dataset.uns["dataset_status_hash"]
    assert mock_dataset.uns["gating_cols"] is not None
    assert isinstance(mock_dataset.uns["gating_cols"], pd.Index)






        



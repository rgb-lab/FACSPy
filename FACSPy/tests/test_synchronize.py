import pytest
import os

from anndata import AnnData
import FACSPy as fp
from FACSPy.synchronization._synchronize import _dataset_has_been_modified
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.exceptions._exceptions import DataModificationWarning

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
                             subsample_fcs_to = 300)

def test_datamodification_warning(mock_dataset: AnnData):
    mock_dataset = mock_dataset[0:mock_dataset.shape[0]-10]
    with pytest.warns(DataModificationWarning):
        fp.sync.synchronize_dataset(mock_dataset)
    assert not _dataset_has_been_modified(mock_dataset)

def test_rehashing(mock_dataset: AnnData):
    """datasets should not be modified after syncing"""
    mock_dataset = mock_dataset[0:mock_dataset.shape[0]-10]
    fp.sync.synchronize_dataset(mock_dataset)
    assert not _dataset_has_been_modified(mock_dataset)

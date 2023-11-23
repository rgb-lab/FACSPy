import pytest
import os

from anndata import AnnData
import FACSPy as fp
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.synchronization._hash_generation import (generate_dataset_obs_hash,
                                                     generate_dataset_var_hash,
                                                     generate_hash_dict)

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
                             workspace = workspace)

def test_generate_hash_dict(mock_dataset: AnnData):
    x = generate_hash_dict(mock_dataset)
    assert isinstance(x, dict)
    assert x["obs_names"] == generate_dataset_obs_hash(mock_dataset)
    assert x["var_names"] == generate_dataset_var_hash(mock_dataset)




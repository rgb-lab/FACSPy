import pytest
import os
import FACSPy as fp
from anndata import AnnData
import numpy as np
from FACSPy.dataset._utils import (asinh,
                                   transform_data_array,
                                   get_control_samples,
                                   get_stained_samples,
                                   create_sample_subset_with_controls,
                                   find_corresponding_control_samples)

from FACSPy.dataset._supplements import Metadata, Panel, CofactorTable
from FACSPy.dataset._workspaces import FlowJoWorkspace


def test_transform_data_array():
    cofactors = np.array([1,1,1], dtype = np.float64)
    input_data = np.array([1,1,1], dtype = np.float64)
    x = transform_data_array(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)

    cofactors = [5,5,5]
    input_data = np.array([5,5,5])
    x = transform_data_array(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)

def test_asinh_transformation():
    cofactors = [1,1,1]
    input_data = np.array([1,1,1])
    x = asinh(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)

    cofactors = [5,5,5]
    input_data = np.array([5,5,5])
    x = asinh(input_data, cofactors)
    np.testing.assert_array_almost_equal(np.repeat([0.88137359],3), x)

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


def test_get_stained_samples(mock_dataset: AnnData):
    stained_samples = get_stained_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "sample_ID")
    assert stained_samples == ["1", "3", "5"]
    stained_samples = get_stained_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "file_name")
    assert stained_samples == ["file1_stained.fcs", "file2_stained.fcs", "file3_stained.fcs"]


def test_get_control_samples(mock_dataset: AnnData):
    stained_samples = get_control_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "sample_ID")
    assert stained_samples == ["2", "4", "6"]
    stained_samples = get_control_samples(mock_dataset.uns["metadata"].dataframe,
                                          by = "file_name")
    assert stained_samples == ["file1_unstained.fcs", "file2_unstained.fcs", "file3_unstained.fcs"]

def test_find_corresponding_control_samples(mock_dataset: AnnData):
    ccs = find_corresponding_control_samples(mock_dataset,
                                             by = "file_name")
    assert ccs[0] == ["file1_stained.fcs", "file2_stained.fcs", "file3_stained.fcs"]
    assert isinstance(ccs[1], dict)
    ccs_dict = ccs[1]
    assert "file1_stained.fcs" in ccs_dict.keys()
    assert "file2_stained.fcs" in ccs_dict.keys()
    assert "file3_stained.fcs" in ccs_dict.keys()
    assert ccs_dict["file1_stained.fcs"] == ["file1_unstained.fcs"]
    assert ccs_dict["file2_stained.fcs"] == ["file2_unstained.fcs"]
    assert ccs_dict["file3_stained.fcs"] == ["file3_unstained.fcs"]

def test_create_sample_subset_with_controls(mock_dataset: AnnData):
    ccs = find_corresponding_control_samples(mock_dataset,
                                             by = "file_name")
    x = create_sample_subset_with_controls(mock_dataset,
                                           "file1_stained.fcs",
                                           corresponding_controls = ccs[1],
                                           match_cell_number = False)
    assert x.shape == (99351, 21)
    assert len(x.obs["file_name"].unique()) == 2

def test_create_sample_subset_with_controls_matching(mock_dataset: AnnData):
    ccs = find_corresponding_control_samples(mock_dataset,
                                             by = "file_name")
    x = create_sample_subset_with_controls(mock_dataset,
                                           "file1_stained.fcs",
                                           corresponding_controls = ccs[1],
                                           match_cell_number = True)
    assert x.shape == (68116, 21)
    assert len(x.obs["file_name"].unique()) == 2


    

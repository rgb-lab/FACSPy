import pytest
import pandas as pd
from ..dataset.supplements import Panel, Metadata
from ..exceptions.exceptions import (SupplementDataTypeError,
                                     SupplementFileNotFoundError,
                                     SupplementCreationError,
                                     SupplementColumnError)
import os

@pytest.fixture
def mock_metadata_correct():
    return pd.DataFrame(
        {
            "sample_id": list(range(2)),
            "file_name": ["some_file.fcs", "some_other_file.fcs"]
        }
    )

@pytest.fixture
def mock_metadata_wrong_colnames():
    return pd.DataFrame(
        {
            "sample_ideas": list(range(2)),
            "file_namez": ["some_file.fcs", "some_other_file.fcs"]
        }
    )
    

@pytest.fixture
def mock_metadata_np_array_instead_of_dataframe():
    df = pd.DataFrame(
        {
            "sample_id": list(range(2)),
            "file_name": ["some_file.fcs", "some_other_file.fcs"]
        }
    )
    return df.to_numpy()

@pytest.fixture
def mock_panel_correct():
    return pd.DataFrame(
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

@pytest.fixture
def mock_panel_nparray_instead_of_dataframe():
    df = pd.DataFrame({"fcs_colname": ["FSC-A", "SSC-A", "APC-H7-A", "FJComp-PE-CF594-A", "Comp-PE-A"],
                        "antigens": ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"]})
    return df.to_numpy()

@pytest.fixture
def mock_panel_wrong_colnames():
    return pd.DataFrame(
        {
            "names": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "antigenitos": ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"],
        }
    )

def test_metadata_correct(mock_metadata_correct):
    x = Metadata(metadata = mock_metadata_correct)
    assert isinstance(x, Metadata)

def test_panel_correct(mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    assert isinstance(x, Panel)


def test_metadata_from_fcs():
    x = Metadata(from_fcs = True)
    assert x.dataframe.shape == (0,2)
    assert x.source == "read from fcs"

def test_panel_from_fcs():
    x = Panel(from_fcs = True)
    assert x.dataframe.shape == (0,2) # only two column names
    assert x.source == "read from fcs"


def test_panel_input_parameters(mock_panel_nparray_instead_of_dataframe):
    with pytest.raises(SupplementDataTypeError):
        _ = Panel(panel = mock_panel_nparray_instead_of_dataframe)

def test_metadata_input_parameters(mock_metadata_np_array_instead_of_dataframe):
    with pytest.raises(SupplementDataTypeError):
        _ = Metadata(metadata = mock_metadata_np_array_instead_of_dataframe)


def test_panel_wrong_colnames(mock_panel_wrong_colnames):
    with pytest.raises(SupplementColumnError):
        _ = Panel(panel = mock_panel_wrong_colnames)

def test_metadata_wrong_colnames(mock_metadata_wrong_colnames):
    with pytest.raises(SupplementColumnError):
        _ = Metadata(metadata = mock_metadata_wrong_colnames)


def test_panel_creation_errors():
    with pytest.raises(SupplementCreationError):
        _ = Panel()
    with pytest.raises(SupplementFileNotFoundError):
        _ = Panel(file_name = "panel.txt")
    with pytest.raises(SupplementFileNotFoundError):
        _ = Panel(file_name = "panel.txt", input_directory = os.getcwd())

def test_metadata_creation_errors():
    with pytest.raises(SupplementCreationError):
        _ = Metadata()
    with pytest.raises(SupplementFileNotFoundError):
        _ = Metadata(file_name = "metadata.txt")
    with pytest.raises(SupplementFileNotFoundError):
        _ = Metadata(file_name = "metadata.txt", input_directory = os.getcwd())






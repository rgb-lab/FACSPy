import pytest
import pandas as pd
from ..dataset.supplements import Panel, Metadata, CofactorTable
from ..exceptions.exceptions import (SupplementDataTypeError,
                                     SupplementFileNotFoundError,
                                     SupplementCreationError,
                                     SupplementColumnError)
import os


@pytest.fixture
def mock_cofactors_correct():
    return pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "cofactors": list(range(5)),
        }
    )

@pytest.fixture
def mock_cofactors_np_array_instead_of_dataframe():
    df = pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "cofactors": list(range(5)),
        }
    )
    return df.to_numpy()

@pytest.fixture
def mock_cofactors_wrong_colname():
    return pd.DataFrame(
        {
            "fcs_colnamez": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A",
            ],
            "cofactor": list(range(5)),
        }
    )

@pytest.fixture
def mock_metadata_correct():
    return pd.DataFrame(
        {
            "sample_ID": list(range(2)),
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

def test_cofactor_correct(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    assert isinstance(x, CofactorTable)


def test_metadata_from_fcs():
    x = Metadata(from_fcs = True)
    assert x.dataframe.shape == (0,2)
    assert x.source == "read from fcs"

def test_panel_from_fcs():
    x = Panel(from_fcs = True)
    assert x.dataframe.shape == (0,2) # only two column names
    assert x.source == "read from fcs"

def test_cofactors_from_fcs():
    x = CofactorTable(from_fcs = True)
    assert x.dataframe.shape == (0,2) # only two column names
    assert x.source == "read from fcs"

def test_panel_input_parameters(mock_panel_nparray_instead_of_dataframe):
    with pytest.raises(SupplementDataTypeError):
        _ = Panel(panel = mock_panel_nparray_instead_of_dataframe)

def test_metadata_input_parameters(mock_metadata_np_array_instead_of_dataframe):
    with pytest.raises(SupplementDataTypeError):
        _ = Metadata(metadata = mock_metadata_np_array_instead_of_dataframe)

def test_cofactor_input_parameters(mock_cofactors_np_array_instead_of_dataframe):
    with pytest.raises(SupplementDataTypeError):
        _ = CofactorTable(cofactors = mock_cofactors_np_array_instead_of_dataframe)


def test_panel_wrong_colnames(mock_panel_wrong_colnames):
    with pytest.raises(SupplementColumnError):
        _ = Panel(panel = mock_panel_wrong_colnames)

def test_metadata_wrong_colnames(mock_metadata_wrong_colnames):
    with pytest.raises(SupplementColumnError):
        _ = Metadata(metadata = mock_metadata_wrong_colnames)

def test_cofactors_wrong_colnames(mock_cofactors_wrong_colname):
    with pytest.raises(SupplementColumnError):
        _ = CofactorTable(cofactors = mock_cofactors_wrong_colname)



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

def test_cofactors_creation_errors():
    with pytest.raises(SupplementCreationError):
        _ = CofactorTable()
    with pytest.raises(SupplementFileNotFoundError):
        _ = CofactorTable(file_name = "cofactors.txt")
    with pytest.raises(SupplementFileNotFoundError):
        _ = CofactorTable(file_name = "cofactors.txt", input_directory = os.getcwd())







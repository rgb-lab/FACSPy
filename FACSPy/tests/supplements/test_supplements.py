import pytest
import pandas as pd
from .fixtures import (mock_cofactors_correct,
                       mock_metadata_correct,
                       mock_panel_correct,
                       mock_cofactors_wrong_colname,
                       mock_metadata_wrong_colnames,
                       mock_panel_wrong_colnames,
                       mock_cofactors_np_array_instead_of_dataframe,
                       mock_metadata_np_array_instead_of_dataframe,
                       mock_panel_nparray_instead_of_dataframe,
                       mock_panel_with_prefixes,
                       mock_cofactors_with_prefixes)
from ...dataset.supplements import Panel, Metadata, CofactorTable
from ...exceptions.supplements import (SupplementDataTypeError,
                                      SupplementFileNotFoundError,
                                      SupplementCreationError,
                                      SupplementColumnError)
import os


### Base Supplement tests
def test_save_to_file_metadata_reread_with_pd(tmp_path, mock_metadata_correct):
    x = Metadata(metadata = mock_metadata_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_metadata.csv"))
    df = pd.read_csv(os.path.join(tmp_path, "some_metadata.csv"))
    assert "sample_ID" in df.columns
    assert "file_name" in df.columns
    assert len(df) == 2

def test_save_to_file_panel_reread_with_pd(tmp_path, mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_panel.csv"))
    df = pd.read_csv(os.path.join(output_directory, "some_panel.csv"))
    assert "fcs_colname" in df.columns
    assert "antigens" in df.columns
    assert len(df) == 5

def test_save_to_file_cofactors_reread_with_pd(tmp_path, mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_cofactors.csv"))
    df = pd.read_csv(os.path.join(output_directory, "some_cofactors.csv"))
    assert "fcs_colname" in df.columns
    assert "cofactors" in df.columns
    assert len(df) == 5

def test_save_to_file_alternative_file_ending(tmp_path, mock_panel_correct):
    x = Panel(panel=mock_panel_correct)
    output_directory = tmp_path
    x.write(output_directory=os.path.join(output_directory, "some_panel.txt"))
    df = pd.read_csv(os.path.join(output_directory, "some_panel.txt"))
    assert "fcs_colname" in df.columns
    assert "antigens" in df.columns

def test_save_to_file_metadata_reread_with_fp(tmp_path, mock_metadata_correct):
    x = Metadata(metadata = mock_metadata_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_metadata.csv"))
    y = Metadata(input_directory = output_directory,
                 file_name = "some_metadata.csv")
    assert "sample_ID" in y.dataframe.columns
    assert "file_name" in y.dataframe.columns
    assert len(y.dataframe) == 2 

def test_save_to_file_panel_reread_with_fp(tmp_path, mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_panel.csv"))
    y = Panel(input_directory = output_directory,
              file_name = "some_panel.csv")
    assert "fcs_colname" in y.dataframe.columns
    assert "antigens" in y.dataframe.columns
    assert len(y.dataframe) == 5

def test_save_to_file_cofactors_reread_with_fp(tmp_path, mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_cofactors.csv"))
    y = CofactorTable(input_directory = output_directory,
                      file_name = "some_cofactors.csv")
    assert "fcs_colname" in y.dataframe.columns
    assert "cofactors" in y.dataframe.columns
    assert len(y.dataframe) == 5

def test_metadata_correct(mock_metadata_correct):
    x = Metadata(metadata = mock_metadata_correct)
    assert isinstance(x, Metadata)

def test_panel_correct(mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    assert isinstance(x, Panel)

def test_cofactor_correct(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    assert isinstance(x, CofactorTable)

def test_strip_prefixes_in_panel(mock_panel_with_prefixes):
    x = Panel(panel = mock_panel_with_prefixes)
    assert all(
        k not in x.dataframe["fcs_colname"].to_list()
        for k in ["Comp", "FJComp"]
    )

def test_strip_prefixes_in_cofactor_table(mock_cofactors_with_prefixes):
    x = CofactorTable(cofactors = mock_cofactors_with_prefixes)
    assert all(
        k not in x.dataframe["fcs_colname"].to_list()
        for k in ["Comp", "FJComp"]
    )

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







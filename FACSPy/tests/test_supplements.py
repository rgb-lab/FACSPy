import pytest
import pandas as pd
import os

from FACSPy.dataset._supplements import Panel, Metadata, CofactorTable
from FACSPy.exceptions._supplements import (SupplementInputTypeError,
                                           SupplementFileNotFoundError,
                                           SupplementCreationError,
                                           SupplementColumnError,
                                           SupplementNoInputDirectoryError)

### Fixtures

@pytest.fixture
def mock_panel_with_prefixes():
    return pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "FSC-H",
                "APC-H7-A",
                "FJComp-PE-CF594-A",
                "Comp-PE-A"
            ],
            "antigens": [
                "FSC-A",
                "SSC-A",
                "CD16",
                "Live/Dead",
                "mitoTracker"
            ]

        }
    )

@pytest.fixture
def mock_cofactors_with_prefixes():
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
def mock_metadata_with_factors():
    return pd.DataFrame(
        {
            "sample_ID": list(range(5)),
            "file_name": [f"file{i}" for i in range(5)],
            "some": ["some" for _ in range(5)],
            "metadata": ["metadata" for _ in range(5)],
            "factors": ["factors" for _ in range(5)]
        }
    )

@pytest.fixture
def mock_cofactors_correct():
    return pd.DataFrame(
        {
            "fcs_colname": [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "PE-CF594-A",
                "PE-A",
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
                "PE-CF594-A",
                "PE-A",
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

def test_metadata_from_fcs_wo_input_dir():
    with pytest.raises(SupplementNoInputDirectoryError):
        x = Metadata(from_fcs = True)

def test_metadata_from_fcs(tmp_path):
    x = Metadata(input_directory = tmp_path,
                 from_fcs = True)
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
    with pytest.raises(SupplementInputTypeError):

        _ = Panel(panel = mock_panel_nparray_instead_of_dataframe)

def test_metadata_input_parameters(mock_metadata_np_array_instead_of_dataframe):
    with pytest.raises(SupplementInputTypeError):
        _ = Metadata(metadata = mock_metadata_np_array_instead_of_dataframe)

def test_cofactor_input_parameters(mock_cofactors_np_array_instead_of_dataframe):
    with pytest.raises(SupplementInputTypeError):
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

def test_metadata_factor_extraction(mock_metadata_with_factors):
    x = Metadata(metadata = mock_metadata_with_factors)
    assert "some" in x.factors
    assert "metadata" in x.factors
    assert "factors" in x.factors

def test_metadata_factor_extraction_from_function(mock_metadata_with_factors):
    x = Metadata(metadata = mock_metadata_with_factors)
    md_factors = x.get_factors()
    assert "some" in md_factors
    assert "metadata" in md_factors
    assert "factors" in md_factors

def test_metadata_categoricals(mock_metadata_with_factors):
    x = Metadata(metadata = mock_metadata_with_factors)
    assert x.dataframe["sample_ID"].dtype.name == "category"
    assert x.dataframe["file_name"].dtype.name == "category"
    assert x.dataframe["some"].dtype.name == "category"
    assert x.dataframe["metadata"].dtype.name == "category"
    assert x.dataframe["factors"].dtype.name == "category"


def test_panel_get_antigens(mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    assert x.get_antigens() == ["FSC-A", "SSC-A", "CD16", "Live/Dead", "mitoTracker"]

def test_panel_get_channels(mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    assert x.get_channels() == [
                "FSC-A",
                "SSC-A",
                "APC-H7-A",
                "PE-CF594-A",
                "PE-A",
            ]

def test_cofactors_get_cofactor(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    assert x.get_cofactor("APC-H7-A") == 2
    assert x.get_cofactor("PE-A") == 4

def test_cofactors_set_cofactor(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.set_cofactor("APC-H7-A", 200)
    assert x.get_cofactor("APC-H7-A") == 200

def test_cofactor_set_columns(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.set_columns([str(i) for i in range(5)])
    assert x.dataframe["fcs_colname"].to_list() == [str(i) for i in range(5)]

def test_cofactors_set_cofactors(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.set_cofactors([200 for _ in range(5)])
    assert x.get_cofactor("APC-H7-A") == 200
    y = CofactorTable(cofactors = mock_cofactors_correct)
    y.set_cofactors(cytof = True)
    assert y.get_cofactor("APC-H7-A") == 5
    with pytest.raises(ValueError):
        y.set_cofactors()




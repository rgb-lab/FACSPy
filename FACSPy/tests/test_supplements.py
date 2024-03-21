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
                "CD16",
                "Live/Dead",
                "mitoTracker"

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
                "CD16",
                "Live/Dead",
                "mitoTracker"
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
                "CD16",
                "Live/Dead",
                "mitoTracker"
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
                "CD16",
                "Live/Dead",
                "mitoTracker"
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
# .write() method
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
    y = Metadata(os.path.join(output_directory, "some_metadata.csv"))
    assert "sample_ID" in y.dataframe.columns
    assert "file_name" in y.dataframe.columns
    assert len(y.dataframe) == 2 

def test_save_to_file_panel_reread_with_fp(tmp_path, mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_panel.csv"))
    y = Panel(os.path.join(output_directory, "some_panel.csv"))
    assert "fcs_colname" in y.dataframe.columns
    assert "antigens" in y.dataframe.columns
    assert len(y.dataframe) == 5

def test_save_to_file_cofactors_reread_with_fp(tmp_path, mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    output_directory = tmp_path
    x.write(output_directory = os.path.join(output_directory, "some_cofactors.csv"))
    y = CofactorTable(os.path.join(output_directory, "some_cofactors.csv"))
    assert "fcs_colname" in y.dataframe.columns
    assert "cofactors" in y.dataframe.columns
    assert len(y.dataframe) == 5

# ._strip_prefixes method
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

# .to_df() method
def test_to_df_metadata(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    df = x.to_df()
    assert isinstance(df, pd.DataFrame)

def test_to_df_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel= mock_panel_correct)
    df = x.to_df()
    assert isinstance(df, pd.DataFrame)

def test_to_df_cofactors(mock_cofactors_correct: pd.DataFrame):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    df = x.to_df()
    assert isinstance(df, pd.DataFrame)

# ._open_from_file_method
def test_open_from_file_metadata(tmp_path,
                                 mock_metadata_correct: pd.DataFrame):
    file_path = os.path.join(tmp_path, "file.csv")
    mock_metadata_correct.to_csv(file_path, index = False)
    x = Metadata(metadata = mock_metadata_correct)
    df = x._open_from_file(file_path)
    df["sample_ID"] = df["sample_ID"].astype(str)
    df[df.columns] = df[df.columns].astype("category")
    assert isinstance(df, pd.DataFrame)
    assert x.dataframe.equals(df)

def test_open_from_file_panel(tmp_path,
                              mock_panel_correct: pd.DataFrame):
    file_path = os.path.join(tmp_path, "file.csv")
    mock_panel_correct.to_csv(file_path, index = False)
    x = Panel(panel = mock_panel_correct)
    df = x._open_from_file(file_path)
    # we strip the prefixes as it would be internally and tested above.
    df = x._strip_prefixes(df)
    assert isinstance(df, pd.DataFrame)
    assert x.dataframe.equals(df)

def test_open_from_file_cofactors(tmp_path,
                                  mock_cofactors_correct: pd.DataFrame):
    file_path = os.path.join(tmp_path, "file.csv")
    mock_cofactors_correct.to_csv(file_path, index = False)
    x = CofactorTable(cofactors = mock_cofactors_correct)
    df = x._open_from_file(file_path)
    assert isinstance(df, pd.DataFrame)
    assert x.dataframe.equals(df)

# ._fetch_data_source() method
    
def test_fetch_data_source_metadata(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    res = x._fetch_data_source(file = "a_file",
                               data = None,
                               from_fcs = False)
    assert res == "provided file"

    # data should override file
    res = x._fetch_data_source(file = "a_file",
                               data = "something",
                               from_fcs = False)
    assert res == "provided dataframe"
    res = x._fetch_data_source(file = None,
                               data = "something",
                               from_fcs = False)
    assert res == "provided dataframe"

    # from_fcs shoud override file and data
    res = x._fetch_data_source(file = None,
                               data = None,
                               from_fcs = True)
    assert res == "read from fcs"
    res = x._fetch_data_source(file = None,
                               data = "something",
                               from_fcs = True)
    assert res == "read from fcs"
    res = x._fetch_data_source(file = "a_file",
                               data = "something",
                               from_fcs = True)

    with pytest.raises(SupplementCreationError):
        x._fetch_data_source(file = None,
                             data = None,
                             from_fcs = None)

def test_fetch_data_source_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    res = x._fetch_data_source(file = "a_file",
                               data = None,
                               from_fcs = False)
    assert res == "provided file"

    # data should override file
    res = x._fetch_data_source(file = "a_file",
                               data = "something",
                               from_fcs = False)
    assert res == "provided dataframe"
    res = x._fetch_data_source(file = None,
                               data = "something",
                               from_fcs = False)
    assert res == "provided dataframe"

    # from_fcs shoud override file and data
    res = x._fetch_data_source(file = None,
                               data = None,
                               from_fcs = True)
    assert res == "read from fcs"
    res = x._fetch_data_source(file = None,
                               data = "something",
                               from_fcs = True)
    assert res == "read from fcs"
    res = x._fetch_data_source(file = "a_file",
                               data = "something",
                               from_fcs = True)

    with pytest.raises(SupplementCreationError):
        x._fetch_data_source(file = None,
                             data = None,
                             from_fcs = None)

def test_fetch_data_source_cofactor_table(mock_cofactors_correct: pd.DataFrame):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    res = x._fetch_data_source(file = "a_file",
                               data = None,
                               from_fcs = False)
    assert res == "provided file"

    # data should override file
    res = x._fetch_data_source(file = "a_file",
                               data = "something",
                               from_fcs = False)
    assert res == "provided dataframe"
    res = x._fetch_data_source(file = None,
                               data = "something",
                               from_fcs = False)
    assert res == "provided dataframe"

    # from_fcs shoud override file and data
    res = x._fetch_data_source(file = None,
                               data = None,
                               from_fcs = True)
    assert res == "read from fcs"
    res = x._fetch_data_source(file = None,
                               data = "something",
                               from_fcs = True)
    assert res == "read from fcs"
    res = x._fetch_data_source(file = "a_file",
                               data = "something",
                               from_fcs = True)

    with pytest.raises(SupplementCreationError):
        x._fetch_data_source(file = None,
                             data = None,
                             from_fcs = None)

# ._validate_user_supplied_table
def test_validate_user_supplied_table_metadata(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    # we first check the SupplementInputTypeError
    with pytest.raises(SupplementInputTypeError):
        x._validate_user_supplied_table("something", ["sample_ID", "file_name"])
    df = mock_metadata_correct.copy()
    with pytest.raises(SupplementColumnError):
        x._validate_user_supplied_table(df, ["some_other_column", "sample_ID", "file_name"])
    df = df.drop(["sample_ID"], axis = 1)
    with pytest.raises(SupplementColumnError):
        x._validate_user_supplied_table(df, ["sample_ID", "file_name"])

def test_validate_user_supplied_table_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    # we first check the SupplementInputTypeError
    with pytest.raises(SupplementInputTypeError):
        x._validate_user_supplied_table("something", ["fcs_colname", "antigens"])
    df = mock_panel_correct.copy()
    with pytest.raises(SupplementColumnError):
        x._validate_user_supplied_table(df, ["some_other_column", "fcs_colname", "antigens"])
    df = df.drop(["fcs_colname"], axis = 1)
    with pytest.raises(SupplementColumnError):
        x._validate_user_supplied_table(df, ["fcs_colname", "antigens"])

def test_validate_user_supplied_table_cofactors(mock_cofactors_correct: pd.DataFrame):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    # we first check the SupplementInputTypeError
    with pytest.raises(SupplementInputTypeError):
        x._validate_user_supplied_table("something", ["fcs_colname", "antigens"])
    df = mock_cofactors_correct.copy()
    with pytest.raises(SupplementColumnError):
        x._validate_user_supplied_table(df, ["some_other_column", "fcs_colname", "antigens"])
    df = df.drop(["fcs_colname"], axis = 1)
    with pytest.raises(SupplementColumnError):
        x._validate_user_supplied_table(df, ["fcs_colname", "antigens"])

# .fetch_data_from_source() 
def test_fetch_data_from_source_metadata(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    x.source = "provided dataframe"
    res = x._fetch_data_from_source(file = "something",
                                    data = pd.DataFrame(),
                                    from_fcs = False)
    assert isinstance(res, pd.DataFrame)
    x.source = "read from fcs"
    res = x._fetch_data_from_source(file = "something",
                                    data = pd.DataFrame(),
                                    from_fcs = True)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (0,2)

def test_fetch_data_from_source_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    x.source = "provided dataframe"
    res = x._fetch_data_from_source(file = "something",
                                    data = pd.DataFrame(),
                                    from_fcs = False)
    assert isinstance(res, pd.DataFrame)
    x.source = "read from fcs"
    res = x._fetch_data_from_source(file = "something",
                                    data = pd.DataFrame(),
                                    from_fcs = True)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (0,2)

def test_fetch_data_from_source_cofactors(mock_cofactors_correct: pd.DataFrame):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.source = "provided dataframe"
    res = x._fetch_data_from_source(file = "something",
                                    data = pd.DataFrame(),
                                    from_fcs = False)
    assert isinstance(res, pd.DataFrame)
    x.source = "read from fcs"
    res = x._fetch_data_from_source(file = "something",
                                    data = pd.DataFrame(),
                                    from_fcs = True)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (0,2)

# ._remove_unnamed_columns() method
def test_remove_unnamed_columns_metadata(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    x.dataframe["Unnamed: 0"] = "unnamed_column_value"
    x._remove_unnamed_columns()
    assert "Unnamed: 0" not in x.dataframe.columns

def test_remove_unnamed_columns_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    x.dataframe["Unnamed: 0"] = "unnamed_column_value"
    x._remove_unnamed_columns()
    assert "Unnamed: 0" not in x.dataframe.columns

def test_remove_unnamed_columns_cofactors(mock_cofactors_correct: pd.DataFrame):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.dataframe["Unnamed: 0"] = "unnamed_column_value"
    x._remove_unnamed_columns()
    assert "Unnamed: 0" not in x.dataframe.columns


# .select_channels() method
def test_select_channels(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    x.select_channels(["FSC-A", "CD16"])
    assert x.dataframe.shape == (2,2)
    assert x.dataframe["antigens"].tolist() == ["FSC-A", "CD16"]

# .__init__() methods
def test_metadata_correct(mock_metadata_correct):
    x = Metadata(metadata = mock_metadata_correct)
    assert isinstance(x, Metadata)

def test_panel_correct(mock_panel_correct):
    x = Panel(panel = mock_panel_correct)
    assert isinstance(x, Panel)

def test_cofactor_correct(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    assert isinstance(x, CofactorTable)

def test_metadata_from_fcs_wo_input_dir():
    with pytest.raises(SupplementNoInputDirectoryError):
        x = Metadata(from_fcs = True)

def test_metadata_from_fcs(tmp_path):
    x = Metadata(tmp_path,
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

# high-level creation with wrong input parameters
def test_panel_input_parameters(mock_panel_nparray_instead_of_dataframe):
    with pytest.raises(SupplementInputTypeError):
        _ = Panel(panel = mock_panel_nparray_instead_of_dataframe)

def test_metadata_input_parameters(mock_metadata_np_array_instead_of_dataframe):
    with pytest.raises(SupplementInputTypeError):
        _ = Metadata(metadata = mock_metadata_np_array_instead_of_dataframe)

def test_cofactor_input_parameters(mock_cofactors_np_array_instead_of_dataframe):
    with pytest.raises(SupplementInputTypeError):
        _ = CofactorTable(cofactors = mock_cofactors_np_array_instead_of_dataframe)

# high-level creation with wrong colnames
def test_panel_wrong_colnames(mock_panel_wrong_colnames):
    with pytest.raises(SupplementColumnError):
        _ = Panel(panel = mock_panel_wrong_colnames)

def test_metadata_wrong_colnames(mock_metadata_wrong_colnames):
    with pytest.raises(SupplementColumnError):
        _ = Metadata(metadata = mock_metadata_wrong_colnames)

def test_cofactors_wrong_colnames(mock_cofactors_wrong_colname):
    with pytest.raises(SupplementColumnError):
        _ = CofactorTable(cofactors = mock_cofactors_wrong_colname)

# creation Errors
def test_panel_creation_errors():
    with pytest.raises(SupplementCreationError):
        _ = Panel()
    with pytest.raises(SupplementFileNotFoundError):
        _ = Panel(file = "panel.txt")

def test_metadata_creation_errors():
    with pytest.raises(SupplementCreationError):
        _ = Metadata()
    with pytest.raises(SupplementFileNotFoundError):
        _ = Metadata(file = "metadata.txt")

def test_cofactors_creation_errors():
    with pytest.raises(SupplementCreationError):
        _ = CofactorTable()
    with pytest.raises(SupplementFileNotFoundError):
        _ = CofactorTable("cofactors.txt")

## panel methods
# .rename_channel() method
def test_rename_channel_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    fcs_colnames = x.dataframe["fcs_colname"].tolist()
    x.rename_channel("FSC-A", "something")
    assert x.dataframe["fcs_colname"].tolist() == ["something"] + fcs_colnames[1:]

# .rename_antigen() method
def test_rename_antigen_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    antigens = x.dataframe["antigens"].tolist()
    x.rename_antigen("FSC-A", "something")
    assert x.dataframe["antigens"].tolist() == ["something"] + antigens[1:]

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

def test_channel_selection_panel(mock_panel_correct: pd.DataFrame):
    x = Panel(panel = mock_panel_correct)
    x.select_channels(["CD16", "mitoTracker"])
    assert x.dataframe.shape == (2,2)
    assert "CD16" in x.dataframe["antigens"].to_list()
    assert "mitoTracker" in x.dataframe["antigens"].to_list()
    assert "FSC-A" not in x.dataframe["antigens"].to_list()

## metadata methods
# .get_factors() method and ._extract_metadata_factors() method
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

# ._make_dataframe_categorical() method
def test_metadata_make_dataframe_categorical(mock_metadata_with_factors):
    # we test if the functionality works when called directly
    x = Metadata(metadata = mock_metadata_with_factors)
    x.dataframe = x.dataframe.astype(str)
    assert x.dataframe.dtypes.unique().tolist() == ["object"]
    x._make_dataframe_categorical()
    print(x.dataframe.dtypes.unique().tolist())
    assert all(isinstance(dtype, pd.CategoricalDtype) for dtype in x.dataframe.dtypes.unique().tolist())

def test_metadata_categoricals(mock_metadata_with_factors):
    # we test if the functionality works internally
    x = Metadata(metadata = mock_metadata_with_factors)
    assert x.dataframe["sample_ID"].dtype.name == "category"
    assert x.dataframe["file_name"].dtype.name == "category"
    assert x.dataframe["some"].dtype.name == "category"
    assert x.dataframe["metadata"].dtype.name == "category"
    assert x.dataframe["factors"].dtype.name == "category"

# ._manage_dtypes() method
def test_metadata_manage_dtypes(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    x.dataframe["sample_ID"] = x.dataframe["sample_ID"].astype(int)
    assert x.dataframe["sample_ID"].dtype.name == "int32"
    x._manage_dtypes()
    assert x.dataframe["sample_ID"].dtype.name == "object"

# .annotate() method
def test_metadata_annotate_typeerror(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    with pytest.raises(TypeError):
        x.annotate(sample_IDs = ["1", "2"], file_names = ["some_file"])

def test_metadata_annotate_filename(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    x.annotate(file_names = "some_file.fcs", column = "new_col", value = "new_val")
    assert "new_col" in x.dataframe.columns
    assert x.dataframe.loc[x.dataframe["file_name"] == "some_file.fcs", "new_col"].iloc[0] == "new_val"
    assert x.dataframe.loc[x.dataframe["file_name"] == "some_other_file.fcs", "new_col"].isna().all()

def test_metadata_annotate_filename_valueerror(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    with pytest.raises(ValueError):
        x.annotate(file_names = "some_file", column = "new_col", value = "new_val")

def test_metadata_annotate_sample_ID(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    x.annotate(sample_IDs = "1", column = "new_col", value = "new_val")
    assert "new_col" in x.dataframe.columns
    assert x.dataframe.loc[x.dataframe["sample_ID"] == "1", "new_col"].iloc[0] == "new_val"
    assert x.dataframe.loc[x.dataframe["sample_ID"] == "2", "new_col"].isna().all()

def test_metadata_annotate_sample_ID_valueerror(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    with pytest.raises(ValueError):
        x.annotate(sample_IDs = "38", column = "new_col", value = "new_val")

# .group_variable() method
def test_metadata_group_variable(mock_metadata_with_factors: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_with_factors)
    x.dataframe["var_to_group"] = list(range(2,7))
    x.group_variable("var_to_group", 2)
    assert "var_to_group_grouped" in x.dataframe.columns
    assert len(x.dataframe["var_to_group_grouped"].unique() == 3)

def test_metadata_group_variable(mock_metadata_with_factors: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_with_factors)
    x.dataframe["var_to_group"] = list(range(2,7))
    x.group_variable("var_to_group", 3)
    assert "var_to_group_grouped" in x.dataframe.columns
    assert len(x.dataframe["var_to_group_grouped"].unique() == 4)

def test_metadata_group_variable_valueerror(mock_metadata_with_factors: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_with_factors)
    x.dataframe["var_to_group"] = [f"a{i}" for i in range(5)]
    with pytest.raises(ValueError):
        x.group_variable("var_to_group", 2)

# .rename_factor() method
def test_metadata_rename(mock_metadata_with_factors: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_with_factors)
    x.rename_column("some", "other")
    assert not "some" in x.dataframe.columns
    assert "other" in x.dataframe.columns
    
# .sanitize_categoricals() method
def test_metadata_categorical_sanitization(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    x.subset("file_name", "some_file.fcs")
    x._sanitize_categoricals()
    assert len(x.dataframe["file_name"].cat.categories) == 1
    assert len(x.dataframe["sample_ID"].cat.categories) == 1

# .subset() method
def test_metadata_subset(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    x.subset("file_name", "some_file.fcs")
    assert x.dataframe.shape == (1,2)

# .rename_factors() method
def test_metadata_rename_factors_with_static_value(mock_metadata_with_factors: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_with_factors)
    x.rename_values(column = "some",
                    replacement = "other")
    assert "some" in x.dataframe.columns
    assert len(x.dataframe["some"].unique()) == 1
    assert x.dataframe["some"].unique().tolist() == ["other"]

def test_metadata_rename_factors_with_list(mock_metadata_with_factors: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_with_factors)
    x.rename_values(column = "some",
                    replacement = ["other" for _ in range(5)])
    assert "some" in x.dataframe.columns
    assert len(x.dataframe["some"].unique()) == 1
    assert x.dataframe["some"].unique().tolist() == ["other"]

def test_metadata_rename_factors_with_dict(mock_metadata_with_factors: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_with_factors)
    x.rename_values(column = "some",
                    replacement = {"some": "other"})
    assert "some" in x.dataframe.columns
    assert len(x.dataframe["some"].unique()) == 1
    assert x.dataframe["some"].unique().tolist() == ["other"]












## CofactorTable tests

# .get_cofactors() method
def test_cofactors_get_cofactor(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    assert x.get_cofactor("CD16") == 2
    assert x.get_cofactor("mitoTracker") == 4

# .set_cofactors() method
def test_cofactors_set_cofactor(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.set_cofactor("CD16", 200)
    assert x.get_cofactor("CD16") == 200

def test_cofactors_set_cofactors(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.set_cofactors([200 for _ in range(5)])
    assert x.get_cofactor("CD16") == 200
    y = CofactorTable(cofactors = mock_cofactors_correct)
    y.set_cofactors(cytof = True)
    assert y.get_cofactor("CD16") == 5
    with pytest.raises(ValueError):
        y.set_cofactors()

# .set_columns() method
def test_cofactor_set_columns(mock_cofactors_correct):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.set_columns([str(i) for i in range(5)])
    assert x.dataframe["fcs_colname"].to_list() == [str(i) for i in range(5)]

# .channel_selection() method
def test_channel_selection_cofactors(mock_cofactors_correct: pd.DataFrame):
    x = CofactorTable(cofactors = mock_cofactors_correct)
    x.select_channels(["CD16", "mitoTracker"])
    assert x.dataframe.shape == (2,2)
    assert "CD16" in x.dataframe["fcs_colname"].to_list()
    assert "mitoTracker" in x.dataframe["fcs_colname"].to_list()
    assert "FSC-A" not in x.dataframe["fcs_colname"].to_list()

# other
def test_channel_selection_metadata(mock_metadata_correct: pd.DataFrame):
    x = Metadata(metadata = mock_metadata_correct)
    with pytest.raises(TypeError):
        x.select_channels(["CD16", "mitoTracker"])








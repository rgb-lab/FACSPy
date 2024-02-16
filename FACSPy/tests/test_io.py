import pytest
from anndata import AnnData
import pandas as pd
import os

import FACSPy as fp
from FACSPy.io._io import (_make_obs_valid,
                           _make_var_valid,
                           save_dataset,
                           read_dataset)
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
def mock_dataset() -> AnnData:
    input_directory, panel, metadata, workspace = create_supplement_objects()
    adata = fp.create_dataset(input_directory = input_directory,
                              panel = panel,
                              metadata = metadata,
                              workspace = workspace)
    adata.layers["transformed"] = adata.layers["compensated"].copy()
    fp.tl.mfi(adata)
    return adata

def test_make_var_valid(mock_dataset: AnnData):
    mock_dataset.var["some_floats"] = 1.0
    mock_dataset.var["datetimes"] = 2
    mock_dataset.var["datetimes"] = pd.to_datetime(mock_dataset.var["datetimes"]).astype("category")
    assert mock_dataset.var["datetimes"].cat.categories.dtype != "O"
    assert isinstance(mock_dataset.var["datetimes"].cat.categories, pd.DatetimeIndex)
    assert not isinstance(mock_dataset.var["some_floats"].dtype, pd.CategoricalDtype)
    

    _make_var_valid(mock_dataset)
    assert mock_dataset.var["some_floats"].dtype == "O"
    assert mock_dataset.var["datetimes"].cat.categories.dtype == "O"

def test_make_obs_valid(mock_dataset: AnnData):
    mock_dataset.obs["some_floats"] = 1.0
    mock_dataset.obs["datetimes"] = 2
    mock_dataset.obs["datetimes"] = pd.to_datetime(mock_dataset.obs["datetimes"]).astype("category")
    assert mock_dataset.obs["datetimes"].cat.categories.dtype != "O"
    assert isinstance(mock_dataset.obs["datetimes"].cat.categories, pd.DatetimeIndex)
    assert not isinstance(mock_dataset.obs["some_floats"].dtype, pd.CategoricalDtype)
    

    _make_obs_valid(mock_dataset)
    assert mock_dataset.obs["some_floats"].dtype == "float64"
    assert mock_dataset.obs["datetimes"].cat.categories.dtype == "O"

def test_save_dataset(tmpdir,
                      mock_dataset):
    file_name = "test"
    save_dataset(mock_dataset,
                 output_dir = tmpdir,
                 file_name = file_name,
                 overwrite = False)
    assert os.path.isfile(os.path.join(tmpdir, f"{file_name}.uns"))
    assert os.path.isfile(os.path.join(tmpdir, f"{file_name}.h5ad"))
    
def test_save_dataset_overwrite(tmpdir,
                                mock_dataset):
    save_dataset(mock_dataset,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    with pytest.raises(FileExistsError):
        save_dataset(mock_dataset,
                     output_dir = tmpdir,
                     file_name = "test",
                     overwrite = False)

def test_read_file(tmpdir,
                   mock_dataset):
    save_dataset(mock_dataset,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    x = read_dataset(tmpdir,
                     "test")
    assert isinstance(x, AnnData)
    assert isinstance(x.uns["dataset_status_hash"], dict)

def test_read_file_rehash(tmpdir,
                          mock_dataset):
    """tests if dataset gets rehashed"""
    del mock_dataset.uns["dataset_status_hash"]
    save_dataset(mock_dataset,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    x = read_dataset(tmpdir,
                     "test")
    assert isinstance(x, AnnData)
    assert isinstance(x.uns["dataset_status_hash"], dict)

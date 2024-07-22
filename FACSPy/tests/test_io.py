import pytest
from anndata import AnnData
import pandas as pd
import os

import FACSPy as fp
from FACSPy.io._io import (_make_obs_valid,
                           _make_var_valid,
                           save_dataset,
                           read_dataset)

def test_make_var_valid(mock_dataset_mfi_calc: AnnData):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    mock_dataset_mfi_calc.var["some_floats"] = 1.0
    mock_dataset_mfi_calc.var["datetimes"] = 2
    mock_dataset_mfi_calc.var["datetimes"] = pd.to_datetime(mock_dataset_mfi_calc.var["datetimes"]).astype("category")
    assert mock_dataset_mfi_calc.var["datetimes"].cat.categories.dtype != "O"
    assert isinstance(mock_dataset_mfi_calc.var["datetimes"].cat.categories, pd.DatetimeIndex)
    assert not isinstance(mock_dataset_mfi_calc.var["some_floats"].dtype, pd.CategoricalDtype)
    

    _make_var_valid(mock_dataset_mfi_calc)
    assert mock_dataset_mfi_calc.var["some_floats"].dtype == "O"
    assert mock_dataset_mfi_calc.var["datetimes"].cat.categories.dtype == "O"

def test_make_obs_valid(mock_dataset_mfi_calc: AnnData):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    mock_dataset_mfi_calc.obs["some_floats"] = 1.0
    mock_dataset_mfi_calc.obs["datetimes"] = 2
    mock_dataset_mfi_calc.obs["datetimes"] = pd.to_datetime(mock_dataset_mfi_calc.obs["datetimes"]).astype("category")
    assert mock_dataset_mfi_calc.obs["datetimes"].cat.categories.dtype != "O"
    assert isinstance(mock_dataset_mfi_calc.obs["datetimes"].cat.categories, pd.DatetimeIndex)
    assert not isinstance(mock_dataset_mfi_calc.obs["some_floats"].dtype, pd.CategoricalDtype)
    

    _make_obs_valid(mock_dataset_mfi_calc)
    assert mock_dataset_mfi_calc.obs["some_floats"].dtype == "float64"
    assert mock_dataset_mfi_calc.obs["datetimes"].cat.categories.dtype == "O"

def test_save_dataset(tmpdir,
                      mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    file_name = "test"
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = file_name,
                 overwrite = False)
    assert os.path.isfile(os.path.join(tmpdir, f"{file_name}.uns"))
    assert os.path.isfile(os.path.join(tmpdir, f"{file_name}.h5ad"))

def test_save_dataset_deprecation_warning(tmpdir, mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    file_name = "test"
    with pytest.warns(DeprecationWarning):
        save_dataset(mock_dataset_mfi_calc,
                     output_dir = tmpdir,
                     file_name = file_name,
                     overwrite = False)

def test_save_dataset_full_file_path(tmpdir, mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    file_name = "test"
    save_dataset(mock_dataset_mfi_calc,
                 file_name = os.path.join(tmpdir, file_name),
                 overwrite = False)
    assert os.path.isfile(os.path.join(tmpdir, f"{file_name}.uns"))
    assert os.path.isfile(os.path.join(tmpdir, f"{file_name}.h5ad"))

def test_save_dataset_with_file_extension(tmpdir, mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    file_name = "test.h5ad"
    short_name = "test"
    save_dataset(mock_dataset_mfi_calc,
                 file_name = os.path.join(tmpdir, file_name),
                 overwrite = False)
    assert os.path.isfile(os.path.join(tmpdir, f"{short_name}.uns"))
    assert os.path.isfile(os.path.join(tmpdir, f"{short_name}.h5ad"))
    
def test_save_dataset_overwrite(tmpdir,
                                mock_dataset_mfi_calc):
    """setting overwrite to False results in an error."""
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    with pytest.raises(FileExistsError):
        save_dataset(mock_dataset_mfi_calc,
                     output_dir = tmpdir,
                     file_name = "test",
                     overwrite = False)

def test_save_dataset_overwrite_2(tmpdir,
                                  mock_dataset_mfi_calc):
    """setting overwrite to True should work."""
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = True)

def test_read_file_deprecationwarnings(tmpdir,
                                       mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    with pytest.warns(DeprecationWarning):
        x = read_dataset(input_dir = tmpdir,
                         file_name = "test")
    assert isinstance(x, AnnData)
    assert isinstance(x.uns["dataset_status_hash"], dict)

def test_read_file(tmpdir,
                   mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    x = read_dataset(input_dir = tmpdir,
                     file_name = "test")
    assert isinstance(x, AnnData)
    assert isinstance(x.uns["dataset_status_hash"], dict)

def test_read_file_filename_only(tmpdir,
                                 mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    x = read_dataset(file_name = os.path.join(tmpdir, "test"))
    assert isinstance(x, AnnData)
    assert isinstance(x.uns["dataset_status_hash"], dict)

def test_read_file_filename_only_with_extension(tmpdir,
                                                mock_dataset_mfi_calc):
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    x = read_dataset(file_name  = os.path.join(tmpdir, "test.h5ad"))
    assert isinstance(x, AnnData)
    assert isinstance(x.uns["dataset_status_hash"], dict)

def test_read_file_rehash(tmpdir,
                          mock_dataset_mfi_calc):
    """tests if dataset gets rehashed"""
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    del mock_dataset_mfi_calc.uns["dataset_status_hash"]
    save_dataset(mock_dataset_mfi_calc,
                 output_dir = tmpdir,
                 file_name = "test",
                 overwrite = False)
    x = read_dataset(input_dir = tmpdir,
                     file_name = "test")
    assert isinstance(x, AnnData)
    assert isinstance(x.uns["dataset_status_hash"], dict)

def test_read_dataset_ValueError(mock_dataset_mfi_calc):
    # defaults are input_dir = None and file_name = None
    mock_dataset_mfi_calc = mock_dataset_mfi_calc.copy()
    with pytest.raises(ValueError):
        fp.read_dataset()

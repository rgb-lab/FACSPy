import pytest

from FACSPy.dataset._transformation import transform
from FACSPy.exceptions.exceptions import InvalidTransformationError
from FACSPy.exceptions.supplements import SupplementFormatError
from FACSPy.dataset._supplements import CofactorTable

import anndata as ad
from anndata import AnnData
import numpy as np
import pandas as pd


@pytest.fixture
def mock_anndata():
    return ad.AnnData(
        X = np.zeros((10,10), dtype = np.float64),
        var = pd.DataFrame(data = np.zeros(shape = (10,2)).astype(str),
                           columns = ["pns", "pnn"],
                           index = [f"channel{i}" for i in range(10)]),
        layers = {
            "raw": np.repeat([1], 100).reshape(10,10).astype(np.float64),
            "compensated": np.repeat([10], 100).reshape(10,10).astype(np.float64),
            "negative": np.repeat([-1], 100).reshape(10,10).astype(np.float64)
        }
    )

def test_wrong_cofactor_table(mock_anndata):
    with pytest.raises(SupplementFormatError):
        transform(mock_anndata,
                  transform = "log",
                  cofactor_table = pd.DataFrame())

def test_copy_behaviour(mock_anndata):
    x = transform(mock_anndata,
                  transform = "log",
                  copy = False)
    assert x is None
    x = transform(mock_anndata,
                  transform = "log",
                  copy = True)
    assert isinstance(x, AnnData)

def test_asinh_transform(mock_anndata: AnnData):
    cofactor_table = CofactorTable(cofactors = pd.DataFrame(data = {"fcs_colname": [f"channel{i}" for i in range(10)],
                                                                    "cofactors": 1}))
    transform(mock_anndata,
              transform = "asinh",
              cofactor_table = cofactor_table)
    """
    transform default is "compensated" layer
    """
    wanted_result = np.repeat([2.99822295], 100).reshape(10,10).astype(np.float64)
    np.testing.assert_array_almost_equal(mock_anndata.layers["asinh"], wanted_result)

def test_logicle_transform(mock_anndata: AnnData):
    transform(mock_anndata,
              transform = "logicle")
    """
    transform default is "compensated" layer
    """
    wanted_result = np.repeat([0.12230428], 100).reshape(10,10).astype(np.float64)
    np.testing.assert_array_almost_equal(mock_anndata.layers["logicle"], wanted_result)

def test_logicle_transform_kwargs(mock_anndata: AnnData):
    transform(mock_anndata,
              transform = "logicle",
              transform_kwargs = {"m": 3.5})
    wanted_result = np.repeat([0.14430281], 100).reshape(10,10).astype(np.float64)
    np.testing.assert_array_almost_equal(mock_anndata.layers["logicle"], wanted_result)

def test_hyperlog_transform(mock_anndata: AnnData):
    transform(mock_anndata,
              transform = "hyperlog")
    """
    transform default is "compensated" layer
    """
    wanted_result = np.repeat([0.12082608], 100).reshape(10,10).astype(np.float64)
    np.testing.assert_array_almost_equal(mock_anndata.layers["hyperlog"], wanted_result)

def test_hyperlog_transform_kwargs(mock_anndata: AnnData):
    transform(mock_anndata,
              transform = "hyperlog",
              transform_kwargs = {"m": 3.5})
    wanted_result = np.repeat([0.14414142], 100).reshape(10,10).astype(np.float64)
    np.testing.assert_array_almost_equal(mock_anndata.layers["hyperlog"], wanted_result)

def test_log_transform(mock_anndata: AnnData):
    transform(mock_anndata,
              transform = "log")
    """
    transform default is "compensated" layer
    """
    wanted_result = np.repeat([0.01810224], 100).reshape(10,10).astype(np.float64)
    np.testing.assert_array_almost_equal(mock_anndata.layers["log"], wanted_result)

def test_log_transform_kwargs(mock_anndata: AnnData):
    transform(mock_anndata,
              transform = "log",
              transform_kwargs = {"m": 3.5})
    wanted_result = np.repeat([-0.26243998], 100).reshape(10,10).astype(np.float64)
    np.testing.assert_array_almost_equal(mock_anndata.layers["log"], wanted_result)
    

def test_invalid_transform_error(mock_anndata):
    with pytest.raises(InvalidTransformationError):
        _ = transform(adata = mock_anndata,
                      transform = "whatever")
        
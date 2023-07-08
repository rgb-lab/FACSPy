import pytest
import pandas as pd
import anndata as ad
import numpy as np

from ...utils import equalize_groups

@pytest.fixture
def mock_anndata():
    adata1 = ad.AnnData(
        X = np.zeros((10,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [1] * 10,
                "donor": [1] * 10,
                "something": [1] * 10
            }
        )
    )
    adata2 = ad.AnnData(
        X = np.zeros((100,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [2] * 100,
                "donor": [2] * 100,
                "something": [2] * 100
            }
        )
    )
    adata3 = ad.AnnData(
        X = np.zeros((1000,10)),
        obs = pd.DataFrame(
            data = {
                "sample_ID": [3] * 1000,
                "donor": [3] * 1000,
                "something": [3] * 500 + [4] * 500
            }
        )
    )
    concatenated = ad.concat([adata1, adata2, adata3])
    concatenated.var_names_make_unique()
    concatenated.obs_names_make_unique()
    return concatenated

def test_equalizing_sample_ID(mock_anndata):
    equalize_groups(mock_anndata, fraction = 0.1)
    assert mock_anndata.shape[0] == 3 ## smallest group has 10

def test_equalizing_something(mock_anndata):
    equalize_groups(mock_anndata, n_obs = 10)
    assert mock_anndata.shape[0] == 30


def test_too_much_to_sample_per_group(mock_anndata):
    with pytest.raises(ValueError):
        equalize_groups(mock_anndata, n_obs = 11)

def test_value_error(mock_anndata):
    with pytest.raises(ValueError):
        equalize_groups(mock_anndata, fraction = 0.1, n_obs = 100)

def test_too_much_to_sample(mock_anndata):
    with pytest.raises(ValueError):
        equalize_groups(mock_anndata, n_obs = 10_000)

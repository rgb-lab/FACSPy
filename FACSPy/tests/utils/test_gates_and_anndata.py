import pytest
import anndata as ad
import numpy as np
import pandas as pd
from ...utils import *

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
    concatenated.uns = {"gating_cols": pd.Index(["root{GATE_SEPARATOR}singlets",
                                                 "root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells",
                                                 "root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells{GATE_SEPARATOR}live"])}
    return concatenated


def test_find_gate_path_of_gate(mock_anndata):
    assert find_gate_path_of_gate(mock_anndata, "T_cells") == f"root{GATE_SEPARATOR}singlets"
    assert find_gate_path_of_gate(mock_anndata, "live") == f"root{GATE_SEPARATOR}singlets{GATE_SEPARATOR}T_cells"
    assert find_gate_path_of_gate(mock_anndata, "singlets") == "root"


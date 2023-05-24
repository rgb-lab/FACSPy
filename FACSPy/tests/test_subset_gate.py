import pytest
import anndata as ad
import numpy as np
import pandas as pd

from ..utils import subset_gate

@pytest.fixture
def mock_anndata():
    return ad.AnnData(
        X = np.zeros((7,7), dtype = np.float64),
        var = pd.DataFrame(
            data = {
                "pns": ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "CD3", "time", "CD8"],
                "type": ["scatter", "scatter", "scatter", "scatter", "fluo", "time", "fluo"]
            },
            index = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "BUV 395-A", "Time", "APC-Cy7-A"]
        ),
        uns = {"gating_cols": pd.Index(["root/singlets", "root/singlets/T_cells"])},
        obsm = {"gating": np.array([[1,1,1,1,1,0,0], [1,1,1,0,0,0,0]], dtype = np.float64).T},
        dtype = np.int32
    )

def test_gate_subset_copy_function(mock_anndata):
    subset_gate(mock_anndata,
                gate = "T_cells",
                copy = False)
    assert mock_anndata.shape[0] == 3

def test_gate_subset_copy_function_2(mock_anndata):
    subset_gate(mock_anndata,
                gate = "singlets",
                copy = False)
    assert mock_anndata.shape[0] == 5

def test_gate_subset_return(mock_anndata):
    dataset = subset_gate(mock_anndata,
                          gate = "T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_return(mock_anndata):
    dataset = subset_gate(mock_anndata,
                          gate = "singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_gate_path(mock_anndata):
    dataset = subset_gate(mock_anndata,
                          gate_path = "root/singlets/T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_gate_path(mock_anndata):
    dataset = subset_gate(mock_anndata,
                          gate_path = "root/singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_gate_path_as_gate(mock_anndata):
    dataset = subset_gate(mock_anndata,
                          gate = "root/singlets/T_cells",
                          copy = True)
    assert dataset.shape[0] == 3

def test_gate_subset_gate_path_as_gate(mock_anndata):
    dataset = subset_gate(mock_anndata,
                          gate = "root/singlets",
                          copy = True)
    assert dataset.shape[0] == 5

def test_gate_subset_wrong_inputs(mock_anndata):
    with pytest.raises(TypeError):
        subset_gate(mock_anndata)

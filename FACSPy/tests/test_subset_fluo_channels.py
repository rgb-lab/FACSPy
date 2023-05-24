import pytest
import anndata as ad
import numpy as np
import pandas as pd

from ..utils import subset_fluo_channels

@pytest.fixture
def mock_anndata():
    return ad.AnnData(
        X = np.zeros((7,7)),
        var = pd.DataFrame(
            data = {
                "pns": ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "CD3", "time", "CD8"],
                "type": ["scatter", "scatter", "scatter", "scatter", "fluo", "time", "fluo"]
            },
            index = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "BUV 395-A", "Time", "APC-Cy7-A"]
        )
    )

def test_fluo_channel_copy_function(mock_anndata):
    subset_fluo_channels(mock_anndata,
                         copy = False)
    assert mock_anndata.shape[1] == 2

def test_fluo_channel_subset(mock_anndata):
    dataset = subset_fluo_channels(mock_anndata,
                                   copy = True)
    assert dataset.shape[1] == 2

def test_fluo_channel_subset_2(mock_anndata):
    dataset = subset_fluo_channels(mock_anndata, copy = True)
    assert "BUV 395-A" in dataset.var.index
    assert "APC-Cy7-A" in dataset.var.index
    assert "Time" not in dataset.var.index

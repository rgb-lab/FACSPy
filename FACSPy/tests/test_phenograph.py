import pytest
import os

from anndata import AnnData
import scanpy as sc
import FACSPy as fp

from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.exceptions._exceptions import InvalidScalingError

import phenograph as _phenograph

import numpy as np

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
    sc.pp.subsample(adata, n_obs = 200, random_state = 187)
    return adata

def test_invalid_scaling_error(mock_dataset: AnnData):
    with pytest.raises(InvalidScalingError):
        fp.tl.phenograph(mock_dataset,
                         gate = "live",
                         scaling = "CustomScaler")

def test_save_settings_from_phenograph(mock_dataset: AnnData):
    adata = mock_dataset
    fp.tl.phenograph(adata,
                     gate = "live",
                     layer = "compensated",
                     seed = 187,
                     use_only_fluo = False,
                     scaling = None,
                     exclude = None)
    assert "settings" in adata.uns
    assert "_phenograph_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_phenograph_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"
    assert settings["clustering_algo"] == "leiden"
    assert settings["use_only_fluo"] == False

def test_phenograph_works_as_phenograph(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    phenograph_data = adata.copy()
    phenograph_data.X = phenograph_data.layers["compensated"]
    comms, graph, Q = _phenograph.cluster(phenograph_data.X,
                                          clustering_algo = "leiden",
                                          seed = 187)

    facspy_adata = adata.copy()
    fp.tl.phenograph(facspy_adata,
                     gate = "live",
                     layer = "compensated",
                     use_only_fluo = False,
                     exclude = None,
                     scaling = None,
                     seed = 187,
                     clustering_algo = "leiden")
    assert "live_compensated_phenograph" in facspy_adata.obs
    assert comms.tolist() == facspy_adata.obs["live_compensated_phenograph"].tolist()
    assert (graph != facspy_adata.uns["live_compensated_phenograph_graph"]).nnz == 0
    assert Q == facspy_adata.uns["live_compensated_phenograph_Q"]
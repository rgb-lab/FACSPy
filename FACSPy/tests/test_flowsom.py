import pytest
import os

from anndata import AnnData
import scanpy as sc
import FACSPy as fp

from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.exceptions._exceptions import InvalidScalingError

from FlowSOM import flowsom

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
        fp.tl.flowsom(mock_dataset,
                      gate = "live",
                      scaling = "CustomScaler")

def test_save_settings_from_flowsom(mock_dataset: AnnData):
    adata = mock_dataset
    fp.tl.flowsom(adata,
                  gate = "live",
                  layer = "compensated",
                  random_state = 187,
                  use_only_fluo = False,
                  x_dim = 10,
                  y_dim = 10,
                  sigma = 0.1,
                  scaling = None,
                  exclude = None)
    assert "settings" in adata.uns
    assert "_flowsom_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_flowsom_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"
    assert settings["x_dim"] == 10
    assert settings["y_dim"] == 10
    assert settings["sigma"] == 0.1

def test_flowsom_works_as_flowsom(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    flowsom_adata = adata.copy()
    flowsom_adata.X = flowsom_adata.layers["compensated"]
    
    flowsom_clusters = flowsom(flowsom_adata.X,
                               x_dim = 10,
                               y_dim = 10,
                               random_state = 187)

    facspy_adata = adata.copy()
    fp.tl.flowsom(facspy_adata,
                  gate = "live",
                  layer = "compensated",
                  random_state = 187,
                  use_only_fluo = False,
                  x_dim = 10,
                  y_dim = 10,
                  scaling = None,
                  exclude = None)
    assert "live_compensated_flowsom" in facspy_adata.obs.columns
    assert all(flowsom_clusters == facspy_adata.obs["live_compensated_flowsom"])

def test_flowsom_works_as_flowsom_kwargs(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    flowsom_adata = adata.copy()
    flowsom_adata.X = flowsom_adata.layers["compensated"]
    
    flowsom_clusters = flowsom(flowsom_adata.X,
                               x_dim = 10,
                               y_dim = 10,
                               sigma = 0.1,
                               random_state = 187)

    facspy_adata = adata.copy()
    fp.tl.flowsom(facspy_adata,
                  gate = "live",
                  layer = "compensated",
                  random_state = 187,
                  use_only_fluo = False,
                  sigma = 0.1,
                  x_dim = 10,
                  y_dim = 10,
                  scaling = None,
                  exclude = None)
    assert "live_compensated_flowsom" in facspy_adata.obs.columns
    assert all(flowsom_clusters == facspy_adata.obs["live_compensated_flowsom"])
     
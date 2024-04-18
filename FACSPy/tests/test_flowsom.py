import pytest
import os

from anndata import AnnData
import scanpy as sc
import FACSPy as fp

from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.exceptions._exceptions import InvalidScalingError

from flowsom.models import FlowSOMEstimator

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
                              workspace = workspace,
                              subsample_fcs_to = 100)
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
                  use_only_fluo = False,
                  scaling = None,
                  exclude = None)
    assert "settings" in adata.uns
    assert "_flowsom_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_flowsom_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"

def test_flowsom_works_as_flowsom(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    flowsom_adata = adata.copy()
    flowsom_adata.X = flowsom_adata.layers["compensated"]

    fse = FlowSOMEstimator(seed = 187, n_clusters = 30)
    flowsom_clusters = fse.fit_predict(flowsom_adata.X)

    fse2 = FlowSOMEstimator(seed = 187, n_clusters = 30)
    flowsom_clusters2 = fse2.fit_predict(flowsom_adata.X)

    assert all(flowsom_clusters == flowsom_clusters2)

    facspy_adata = adata.copy()
    fp.tl.flowsom(facspy_adata,
                  gate = "live",
                  layer = "compensated",
                  use_only_fluo = False,
                  cluster_kwargs = {"seed": 187},
                  scaling = None,
                  exclude = None)
    assert "live_compensated_flowsom" in facspy_adata.obs.columns
    assert all(flowsom_clusters == facspy_adata.obs["live_compensated_flowsom"])

def test_flowsom_works_as_flowsom_kwargs(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    flowsom_adata = adata.copy()
    flowsom_adata.X = flowsom_adata.layers["compensated"]

    fse = FlowSOMEstimator(xdim = 5, ydim = 5, seed = 187, n_clusters = 15)
    flowsom_clusters = fse.fit_predict(flowsom_adata.layers["compensated"])

    facspy_adata = adata.copy()
    fp.tl.flowsom(facspy_adata,
                  gate = "live",
                  layer = "compensated",
                  use_only_fluo = False,
                  xdim = 5,
                  ydim = 5,
                  seed = 187,
                  n_clusters = 15,
                  scaling = None,
                  exclude = None)
    assert "live_compensated_flowsom" in facspy_adata.obs.columns
    assert len(flowsom_clusters) == facspy_adata.shape[0]
    assert all(flowsom_clusters == facspy_adata.obs["live_compensated_flowsom"])


def test_decorator_default_gate_and_default_layer(mock_dataset: AnnData):
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.flowsom(mock_dataset)
    assert "live_compensated_flowsom" in mock_dataset.obs.columns

def test_decorator_default_gate_and_default_layer_only_gate_provided(mock_dataset: AnnData):
    fp.settings.default_layer = "compensated"

    fp.tl.flowsom(mock_dataset, gate = "live")
    assert "live_compensated_flowsom" in mock_dataset.obs.columns

def test_decorator_default_gate_and_default_layer_only_layer_provided(mock_dataset: AnnData):
    fp.settings.default_gate = "live"

    fp.tl.flowsom(mock_dataset, layer = "compensated")
    assert "live_compensated_flowsom" in mock_dataset.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias(mock_dataset: AnnData):
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.flowsom(mock_dataset)
    assert "live_compensated_flowsom" in mock_dataset.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg(mock_dataset: AnnData):
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.flowsom(mock_dataset, "my_personal_gate")
    assert "live_compensated_flowsom" in mock_dataset.obs.columns

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg(mock_dataset: AnnData):
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.flowsom(mock_dataset, gate = "my_personal_gate")
    assert "live_compensated_flowsom" in mock_dataset.obs.columns

import pytest
from anndata import AnnData
import scanpy as sc
import FACSPy as fp

from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.exceptions._exceptions import InvalidScalingError

WSP_FILE_PATH = "FACSPy/_resources/"
WSP_FILE_NAME = "test_wsp.wsp"

def create_supplement_objects():
    INPUT_DIRECTORY = "FACSPy/_resources/test_suite_dataset"
    panel = Panel(input_directory = INPUT_DIRECTORY,
                  file_name = "panel.txt")
    metadata = Metadata(input_directory = INPUT_DIRECTORY,
                        file_name = "metadata_test_suite.csv")
    workspace = FlowJoWorkspace(input_directory = INPUT_DIRECTORY,
                                file_name = "test_suite.wsp")
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
        fp.tl.leiden(mock_dataset,
                     gate = "live",
                     scaling = "CustomScaler")

def test_save_settings_from_leiden(mock_dataset: AnnData):
    adata = mock_dataset
    fp.tl.leiden(adata,
                 gate = "live",
                 layer = "compensated",
                 random_state = 187,
                 use_only_fluo = False,
                 resolution = 5,
                 use_rep = "X",
                 scaling = None,
                 exclude = None)
    assert "settings" in adata.uns
    assert "_leiden_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_leiden_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"
    assert settings["resolution"] == 5
    assert settings["use_only_fluo"] == False

def test_leiden_works_as_scanpy(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated")

    sc.tl.leiden(scanpy_adata,
                 neighbors_key = "live_compensated_neighbors")
    leiden_clusters = scanpy_adata.obs["leiden"]

    facspy_adata = adata.copy()
    fp.tl.leiden(facspy_adata,
                 gate = "live",
                 layer = "compensated",
                 random_seed = 187,
                 use_only_fluo = False,
                 scaling = None,
                 exclude = None)
    assert "live_compensated_leiden" in facspy_adata.obs.columns
    assert all(leiden_clusters == facspy_adata.obs["live_compensated_leiden"])

def test_leiden_works_as_scanpy_kwargs(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    fp.tl.neighbors(adata,
                    gate = "live",
                    layer = "compensated")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]

    sc.tl.leiden(scanpy_adata,
                 resolution = 0.1,
                 neighbors_key = "live_compensated_neighbors")
    leiden_clusters = scanpy_adata.obs["leiden"]


    facspy_adata = adata.copy()
    fp.tl.leiden(facspy_adata,
                 gate = "live",
                 layer = "compensated",
                 random_seed = 187,
                 use_only_fluo = False,
                 resolution = 0.1,
                 scaling = None,
                 exclude = None)
    assert "live_compensated_leiden" in facspy_adata.obs.columns
    assert all(leiden_clusters == facspy_adata.obs["live_compensated_leiden"])

def test_leiden_works_as_scanpy_all_scanpy_funcs(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    fp.tl.pca(scanpy_adata, random_state = 187)
    fp.tl.neighbors(scanpy_adata,
                    use_rep = "X_pca",
                    random_state = 187) 
    sc.tl.leiden(scanpy_adata,
                 resolution = 0.1,
                 neighbors_key = "live_compensated_neighbors",
                 random_state = 187)
    leiden_clusters = scanpy_adata.obs["leiden"]
    facspy_adata = adata.copy()
    fp.tl.leiden(facspy_adata,
                 gate = "live",
                 layer = "compensated",
                 random_seed = 187,
                 use_only_fluo = False,
                 resolution = 0.1,
                 scaling = None,
                 exclude = None)
    assert "live_compensated_leiden" in facspy_adata.obs.columns
    assert all(leiden_clusters == facspy_adata.obs["live_compensated_leiden"])


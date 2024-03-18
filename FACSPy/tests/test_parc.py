
import pytest
import os
from anndata import AnnData
import scanpy as sc
import FACSPy as fp

from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.exceptions._exceptions import InvalidScalingError

import parc as _parc

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
        fp.tl.parc(mock_dataset,
                   gate = "live",
                   scaling = "CustomScaler")

def test_save_settings_from_parc(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    fp.tl.parc(adata,
               gate = "live",
               layer = "compensated",
               random_state = 187,
               use_only_fluo = False,
               dist_std_local = 1,
               scaling = None,
               exclude = None)
    assert "settings" in adata.uns
    assert "_parc_live_compensated" in adata.uns["settings"]
    settings = adata.uns["settings"]["_parc_live_compensated"]
    assert settings["gate"] == "live"
    assert settings["layer"] == "compensated"
    assert settings["dist_std_local"] == 1
    assert settings["use_only_fluo"] == False

def test_parc_works_as_parc(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    parc_adata = adata.copy()
    parc_adata.X = parc_adata.layers["compensated"]
    fp.tl.neighbors(parc_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    exclude = None,
                    scaling = None,
                    random_state = 187)
    
    parcer = _parc.PARC(parc_adata.X,
                        neighbor_graph = parc_adata.obsp["live_compensated_neighbors_connectivities"],
                        random_seed = 187)
    parcer.run_PARC()
    parc_clusters = parcer.labels

    facspy_adata = adata.copy()
    fp.tl.parc(facspy_adata,
               gate = "live",
               layer = "compensated",
               random_seed = 187,
               use_only_fluo = False,
               scaling = None,
               exclude = None)
    assert "live_compensated_parc" in facspy_adata.obs.columns
    assert all(parc_clusters == facspy_adata.obs["live_compensated_parc"])

def test_parc_works_as_parc_kwargs(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    fp.tl.neighbors(adata,
                    gate = "live",
                    layer = "compensated")
    parc_adata = adata.copy()
    parc_adata.X = parc_adata.layers["compensated"]
    
    parcer = _parc.PARC(parc_adata.X,
                        n_iter_leiden = 15,
                        neighbor_graph = parc_adata.obsp["live_compensated_neighbors_connectivities"],
                        random_seed = 187)
    parcer.run_PARC()
    parc_clusters = parcer.labels

    facspy_adata = adata.copy()
    fp.tl.parc(facspy_adata,
               gate = "live",
               layer = "compensated",
               random_seed = 187,
               use_only_fluo = False,
               scaling = None,
               exclude = None,
               n_iter_leiden = 15)
    assert "live_compensated_parc" in facspy_adata.obs.columns
    assert all(parc_clusters == facspy_adata.obs["live_compensated_parc"])


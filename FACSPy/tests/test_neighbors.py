import pytest
from anndata import AnnData
import os

import scanpy as sc
import numpy as np

import FACSPy as fp

from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

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

def test_neighbors_same_as_scanpy_use_rep_X(mock_dataset: AnnData):
    adata = mock_dataset
    adata.X = adata.layers["compensated"]
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    facspy_adata = adata.copy()
    sc.pp.neighbors(scanpy_adata,
                    use_rep = "X")
    conns = scanpy_adata.obsp["connectivities"]
    dists = scanpy_adata.obsp["distances"]
    fp.tl.neighbors(facspy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    use_rep = "X")
    
    dists_facspy = facspy_adata.obsp["live_compensated_neighbors_distances"]
    conns_facspy = facspy_adata.obsp["live_compensated_neighbors_connectivities"]

    assert (conns!= conns_facspy).nnz == 0
    assert (dists!= dists_facspy).nnz == 0

def test_neighbors_same_as_scanpy_use_rep_pca(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    adata.X = adata.layers["compensated"]
    scanpy_adata = adata.copy()
    facspy_adata = adata.copy()

    sc.pp.pca(scanpy_adata,
              random_state = 187)
    sc.pp.neighbors(scanpy_adata,
                    use_rep = "X_pca")
    conns = scanpy_adata.obsp["connectivities"]
    dists = scanpy_adata.obsp["distances"]

    fp.tl.pca(facspy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              random_state = 187)
    fp.tl.neighbors(facspy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    use_rep = "X_pca_live_compensated")

    assert np.array_equal(scanpy_adata.obsm["X_pca"],
                          facspy_adata.obsm["X_pca_live_compensated"])
    
    dists_facspy = facspy_adata.obsp["live_compensated_neighbors_distances"]
    conns_facspy = facspy_adata.obsp["live_compensated_neighbors_connectivities"]

    assert (conns != conns_facspy).nnz == 0
    assert (dists != dists_facspy).nnz == 0

def test_neighbors_same_as_scanpy_use_rep_pca_kwarg_passing(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    adata.X = adata.layers["compensated"]
    scanpy_adata = adata.copy()
    facspy_adata = adata.copy()

    sc.pp.pca(scanpy_adata,
              random_state = 187)
    sc.pp.neighbors(scanpy_adata,
                    n_neighbors = 10,
                    use_rep = "X_pca")
    conns = scanpy_adata.obsp["connectivities"]
    dists = scanpy_adata.obsp["distances"]

    fp.tl.pca(facspy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              random_state = 187)
    fp.tl.neighbors(facspy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    use_rep = "X_pca_live_compensated",
                    n_neighbors = 10)

    assert np.array_equal(scanpy_adata.obsm["X_pca"],
                          facspy_adata.obsm["X_pca_live_compensated"])
    
    dists_facspy = facspy_adata.obsp["live_compensated_neighbors_distances"]
    conns_facspy = facspy_adata.obsp["live_compensated_neighbors_connectivities"]

    assert (conns != conns_facspy).nnz == 0
    assert (dists != dists_facspy).nnz == 0
    
def test_neighbors_same_as_scanpy_use_rep_pca_kwarg_passing_2(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    adata.X = adata.layers["compensated"]
    scanpy_adata = adata.copy()
    facspy_adata = adata.copy()

    sc.pp.pca(scanpy_adata,
              random_state = 187)
    sc.pp.neighbors(scanpy_adata,
                    n_neighbors = 10,
                    n_pcs = 10,
                    use_rep = "X_pca")
    conns = scanpy_adata.obsp["connectivities"]
    dists = scanpy_adata.obsp["distances"]
    fp.tl.pca(facspy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              random_state = 187)
    fp.tl.neighbors(facspy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    use_rep = "X_pca_live_compensated",
                    n_neighbors = 10,
                    n_pcs = 10)
    assert np.array_equal(scanpy_adata.obsm["X_pca"],
                          facspy_adata.obsm["X_pca_live_compensated"])
    
    dists_facspy = facspy_adata.obsp["live_compensated_neighbors_distances"]
    conns_facspy = facspy_adata.obsp["live_compensated_neighbors_connectivities"]

    assert (conns != conns_facspy).nnz == 0
    assert (dists != dists_facspy).nnz == 0
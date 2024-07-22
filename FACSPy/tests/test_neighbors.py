from anndata import AnnData

import scanpy as sc
import numpy as np

import FACSPy as fp


def test_neighbors_same_as_scanpy_use_rep_X(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
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

def test_neighbors_same_as_scanpy_use_rep_pca(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
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

def test_neighbors_same_as_scanpy_use_rep_pca_kwarg_passing(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
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
    
def test_neighbors_same_as_scanpy_use_rep_pca_kwarg_passing_2(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
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

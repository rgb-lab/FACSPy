import pytest
import os
import scanpy as sc
from anndata import AnnData
import numpy as np
import FACSPy as fp
from sklearn.manifold import TSNE
from FACSPy.dataset._supplements import Metadata, Panel
from FACSPy.dataset._workspaces import FlowJoWorkspace

from FACSPy.exceptions._exceptions import InvalidScalingError


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

def test_invalid_scaling(mock_dataset):
    with pytest.raises(InvalidScalingError):
        fp.tl.umap(mock_dataset,
                   gate = "live",
                   scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.tsne(mock_dataset,
                   gate = "live",
                   scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.pca(mock_dataset,
                  gate = "live",
                  scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.diffmap(mock_dataset,
                      gate = "live",
                      scaling = "MyCustomScaler")

def test_pca_works_as_sklearn(mock_dataset: AnnData):
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    from sklearn.decomposition import PCA
    pca_ = PCA(n_components = 20,
               random_state = 187,
               svd_solver = "arpack")
    coords = pca_.fit_transform(adata.layers["compensated"])
    variance = pca_.explained_variance_
    variance_ratio = pca_.explained_variance_ratio_

    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              exclude = None,
              scaling = None)
    facspy_coords = adata.obsm["X_pca_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["pca_live_compensated"]
    assert uns_dict["params"] == {"zero_center": True}
    assert np.array_equal(variance, uns_dict["variance"])
    assert np.array_equal(variance_ratio, uns_dict["variance_ratio"])
    assert adata.X is None

def test_pca_works_as_scanpy(mock_dataset: AnnData):
    mock_dataset.X = mock_dataset.layers["compensated"]
    fp.subset_gate(mock_dataset, "live")
    scanpy_adata = mock_dataset.copy()
    facspy_adata = mock_dataset.copy()
    sc.pp.pca(scanpy_adata,
              random_state = 187)
    fp.tl.pca(facspy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              exclude = None,
              scaling = None)
    assert np.array_equal(scanpy_adata.varm["PCs"],
                          facspy_adata.varm["pca_live_compensated"])
    assert np.array_equal(scanpy_adata.obsm["X_pca"],
                          facspy_adata.obsm["X_pca_live_compensated"])

def test_pca_works_as_scanpy_2(mock_dataset: AnnData):
    mock_dataset.X = mock_dataset.layers["compensated"]
    fp.subset_gate(mock_dataset, "live")
    scanpy_adata = mock_dataset.copy()
    facspy_adata = mock_dataset.copy()
    sc.pp.pca(scanpy_adata,
              svd_solver = "randomized",
              random_state = 187)
    fp.tl.pca(facspy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              exclude = None,
              scaling = None,
              svd_solver = "randomized")
    assert np.array_equal(scanpy_adata.varm["PCs"],
                          facspy_adata.varm["pca_live_compensated"])
    
def test_pca_kwargs(mock_dataset: AnnData):
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    from sklearn.decomposition import PCA
    test_pca_ = PCA(n_components = 20,
                    random_state = 187,
                    svd_solver = "arpack")
    test_coords = test_pca_.fit_transform(adata.layers["compensated"])
    pca_ = PCA(n_components = 20,
               random_state = 188,
               svd_solver = "arpack")
    coords = pca_.fit_transform(adata.layers["compensated"])

    assert not np.array_equal(test_coords, coords)

    variance = pca_.explained_variance_
    variance_ratio = pca_.explained_variance_ratio_

    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              exclude = None,
              scaling = None,
              random_state = 188)
    facspy_coords = adata.obsm["X_pca_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["pca_live_compensated"]
    assert uns_dict["params"] == {"zero_center": True}
    assert np.array_equal(variance, uns_dict["variance"])
    assert np.array_equal(variance_ratio, uns_dict["variance_ratio"])
    assert adata.X is None

def test_pca_kwargs_2(mock_dataset: AnnData):
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    from sklearn.decomposition import PCA
    test_pca_ = PCA(n_components = 20,
                    random_state = 188,
                    svd_solver = "arpack")
    test_coords = test_pca_.fit_transform(adata.layers["compensated"])
    pca_ = PCA(n_components = 20,
               random_state = 188,
               svd_solver = "randomized")
    coords = pca_.fit_transform(adata.layers["compensated"])

    assert not np.array_equal(test_coords, coords)
    variance = pca_.explained_variance_
    variance_ratio = pca_.explained_variance_ratio_

    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              exclude = None,
              scaling = None,
              random_state = 188,
              svd_solver = "randomized")
    facspy_coords = adata.obsm["X_pca_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["pca_live_compensated"]
    assert uns_dict["params"] == {"zero_center": True}
    assert np.array_equal(variance, uns_dict["variance"])
    assert np.array_equal(variance_ratio, uns_dict["variance_ratio"])
    assert adata.X is None

def test_tsne_works_as_sklearn(mock_dataset: AnnData):
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 early_exaggeration = 12,
                 learning_rate = 1000,
                 random_state = 187)
    coords = tsne_.fit_transform(adata.layers["compensated"])

    fp.tl.tsne(adata,
               gate = "live",
               layer = "compensated",
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               n_components = 2,
               random_state = 187,
               use_rep = "X")
    facspy_coords = adata.obsm["X_tsne_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["tsne_live_compensated"]
    assert uns_dict["params"]["perplexity"] == 30
    assert uns_dict["params"]["early_exaggeration"] == 12
    assert uns_dict["params"]["learning_rate"] == 1000
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["use_rep"] == "X"
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["n_components"] == 2

    assert adata.X is None


def test_tsne_works_as_scanpy(mock_dataset: AnnData):
    mock_dataset.X = mock_dataset.layers["compensated"]
    fp.subset_gate(mock_dataset, "live")
    scanpy_adata = mock_dataset.copy()
    facspy_adata = mock_dataset.copy()
    sc.tl.tsne(scanpy_adata,
               random_state = 187,
               use_rep = "X")
    fp.tl.tsne(facspy_adata,
               gate = "live",
               layer = "compensated",
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               n_components = 2,
               use_rep = "X")
    assert np.array_equal(scanpy_adata.obsm["X_tsne"],
                          facspy_adata.obsm["X_tsne_live_compensated"])

def test_tsne_works_as_scanpy_2(mock_dataset: AnnData):
    mock_dataset.X = mock_dataset.layers["compensated"]
    fp.subset_gate(mock_dataset, "live")
    scanpy_adata = mock_dataset.copy()
    facspy_adata = mock_dataset.copy()
    sc.tl.tsne(scanpy_adata,
               early_exaggeration = 400,
               use_rep = "X",
               random_state = 187)
    fp.tl.tsne(facspy_adata,
               gate = "live",
               layer = "compensated",
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               n_components = 2,
               early_exaggeration = 400,
               use_rep = "X")
    uns_dict = facspy_adata.uns["tsne_live_compensated"]
    assert uns_dict["params"]["perplexity"] == 30
    assert uns_dict["params"]["early_exaggeration"] == 400
    assert uns_dict["params"]["learning_rate"] == 1000
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["use_rep"] == "X"
    assert uns_dict["params"]["n_components"] == 2
    assert facspy_adata.X is None
    assert np.array_equal(scanpy_adata.obsm["X_tsne"],
                          facspy_adata.obsm["X_tsne_live_compensated"])

def test_tsne_works_as_sklearn_with_rep(mock_dataset: AnnData):
    """tests if rep pca works"""
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None)
    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 early_exaggeration = 12,
                 learning_rate = 1000,
                 random_state = 187)
    coords = tsne_.fit_transform(adata.obsm["X_pca_live_compensated"])

    fp.tl.tsne(adata,
               gate = "live",
               layer = "compensated",
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               n_components = 2,
               early_exaggeration = 12,
               learning_rate = 1000,
               random_state = 187)
    facspy_coords = adata.obsm["X_tsne_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["tsne_live_compensated"]
    assert uns_dict["params"]["perplexity"] == 30
    assert uns_dict["params"]["early_exaggeration"] == 12
    assert uns_dict["params"]["learning_rate"] == 1000
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["n_components"] == 2

    assert adata.X is None

def test_tsne_works_as_sklearn_with_rep_2(mock_dataset: AnnData):
    """tests if the npcs parameter works"""
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None)
    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 early_exaggeration = 12,
                 learning_rate = 1000,
                 random_state = 187)
    coords = tsne_.fit_transform(adata.obsm["X_pca_live_compensated"][:,:15])

    fp.tl.tsne(adata,
               gate = "live",
               layer = "compensated",
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               n_components = 2,
               early_exaggeration = 12,
               learning_rate = 1000,
               random_state = 187,
               n_pcs = 15)
    facspy_coords = adata.obsm["X_tsne_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["tsne_live_compensated"]
    assert uns_dict["params"]["perplexity"] == 30
    assert uns_dict["params"]["early_exaggeration"] == 12
    assert uns_dict["params"]["learning_rate"] == 1000
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["n_components"] == 2

    assert adata.X is None

def test_tsne_choose_rep_value_error(mock_dataset: AnnData):
    adata = mock_dataset
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    with pytest.raises(ValueError):
        fp.tl.tsne(adata,
                   gate = "live",
                   layer = "compensated",
                   use_rep = "X_pca_live_compensated",
                   n_pcs = 30)
        
def test_tsne_rep_does_not_exist(mock_dataset: AnnData):
    adata = mock_dataset
    with pytest.raises(ValueError):
        fp.tl.tsne(adata,
                   gate = "live",
                   layer = "compensated",
                   use_rep = "X_pca_live_compensated",
                   n_pcs = 30)

def test_tsne_kwargs(mock_dataset: AnnData):
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    test_tsne_ = TSNE(n_components = 2,
                      perplexity = 30,
                      early_exaggeration = 12,
                      learning_rate = 1000,
                      random_state = 187)
    test_coords = test_tsne_.fit_transform(adata.layers["compensated"])

    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 early_exaggeration = 400,
                 learning_rate = 1000,
                 random_state = 187)
    coords = tsne_.fit_transform(adata.layers["compensated"])

    assert not np.array_equal(test_coords, coords)

    fp.tl.tsne(adata,
               gate = "live",
               layer = "compensated",
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               random_state = 187,
               use_rep = "X",
               early_exaggeration = 400,
               n_components = 2)
    facspy_coords = adata.obsm["X_tsne_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["tsne_live_compensated"]
    assert uns_dict["params"]["perplexity"] == 30
    assert uns_dict["params"]["early_exaggeration"] == 400
    assert uns_dict["params"]["learning_rate"] == 1000
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["n_components"] == 2

    assert adata.X is None

def test_tsne_kwargs_2(mock_dataset: AnnData):
    fp.subset_gate(mock_dataset, "live")
    adata = mock_dataset.copy()
    test_tsne_ = TSNE(n_components = 2,
                      perplexity = 30,
                      metric = "euclidean",
                      early_exaggeration = 12,
                      learning_rate = 1000,
                      random_state = 187)
    test_coords = test_tsne_.fit_transform(adata.layers["compensated"])

    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 metric = "cosine",
                 early_exaggeration = 12,
                 learning_rate = 1000,
                 random_state = 187)
    coords = tsne_.fit_transform(adata.layers["compensated"])

    assert not np.array_equal(test_coords, coords)

    fp.tl.tsne(adata,
               gate = "live",
               layer = "compensated",
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               random_state = 187,
               metric = "cosine",
               n_components = 2,
               use_rep = "X")
    facspy_coords = adata.obsm["X_tsne_live_compensated"]
    assert np.array_equal(coords, facspy_coords)
    uns_dict = adata.uns["tsne_live_compensated"]
    assert uns_dict["params"]["perplexity"] == 30
    assert uns_dict["params"]["early_exaggeration"] == 12
    assert uns_dict["params"]["learning_rate"] == 1000
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "cosine"
    assert uns_dict["params"]["n_components"] == 2
    assert uns_dict["params"]["use_rep"] == "X"

    assert adata.X is None

def test_umap_same_as_scanpy(mock_dataset: AnnData):
    """
    we first compute everything after one another
    and only change the umap call
    """
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    fp.tl.pca(scanpy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              exclude = None)
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    exclude = None)
    sc.tl.umap(scanpy_adata,
               n_components = 2,
               neighbors_key = "live_compensated_neighbors",
               random_state = 187)
    facspy_adata = adata.copy()
    fp.tl.pca(facspy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              exclude = None)
    fp.tl.neighbors(facspy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    exclude = None)
    fp.tl.umap(facspy_adata,
               gate = "live",
               layer = "compensated",
               n_components = 2,
               random_state = 187,
               use_only_fluo = False,
               exclude = None,
               scaling = None)
    np.testing.assert_array_almost_equal(scanpy_adata.obsm["X_pca_live_compensated"], facspy_adata.obsm["X_pca_live_compensated"])
    assert (scanpy_adata.obsp["live_compensated_neighbors_connectivities"] != facspy_adata.obsp["live_compensated_neighbors_connectivities"]).nnz == 0
    
    assert np.array_equal(scanpy_adata.obsm["X_umap"],
                          facspy_adata.obsm["X_umap_live_compensated"])

def test_umap_same_as_scanpy_no_precompute_pca_neighbors(mock_dataset: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function
    """
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    fp.tl.pca(scanpy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              exclude = None)
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    exclude = None)
    sc.tl.umap(scanpy_adata,
               n_components = 2,
               neighbors_key = "live_compensated_neighbors",
               random_state = 187)
    facspy_adata = adata.copy()
    fp.tl.umap(facspy_adata,
               gate = "live",
               layer = "compensated",
               n_components = 2,
               random_state = 187,
               use_only_fluo = False,
               exclude = None,
               scaling = None)
    np.testing.assert_array_almost_equal(scanpy_adata.obsm["X_pca_live_compensated"], facspy_adata.obsm["X_pca_live_compensated"])
    assert (scanpy_adata.obsp["live_compensated_neighbors_connectivities"] != facspy_adata.obsp["live_compensated_neighbors_connectivities"]).nnz == 0
    
    assert np.array_equal(scanpy_adata.obsm["X_umap"],
                          facspy_adata.obsm["X_umap_live_compensated"])

def test_umap_same_as_scanpy_no_precompute_pca_neighbors_scanpy_funcs(mock_dataset: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function and compare this to the scanpy
    pca and neighbors call
    """
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    sc.pp.pca(scanpy_adata, random_state = 187)
    sc.pp.neighbors(scanpy_adata,
                    use_rep = "X_pca",
                    random_state = 187)
    sc.tl.umap(scanpy_adata,
               n_components = 2,
               random_state = 187)
    facspy_adata = adata.copy()
    fp.tl.umap(facspy_adata,
               gate = "live",
               layer = "compensated",
               n_components = 2,
               random_state = 187,
               use_only_fluo = False,
               exclude = None,
               scaling = None)
    np.testing.assert_array_almost_equal(scanpy_adata.obsm["X_pca"], facspy_adata.obsm["X_pca_live_compensated"])
    assert (scanpy_adata.obsp["connectivities"] != facspy_adata.obsp["live_compensated_neighbors_connectivities"]).nnz == 0
    
    assert np.array_equal(scanpy_adata.obsm["X_umap"],
                          facspy_adata.obsm["X_umap_live_compensated"])

def test_umap_same_as_scanpy_no_precompute_pca_neighbors_scanpy_funcs_kwargs(mock_dataset: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function and compare this to the scanpy
    pca and neighbors call
    """
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    sc.pp.pca(scanpy_adata, random_state = 187, n_comps = 12)
    sc.pp.neighbors(scanpy_adata,
                    use_rep = "X_pca",
                    random_state = 187,
                    n_neighbors = 10)
    sc.tl.umap(scanpy_adata,
               n_components = 2,
               random_state = 187)
    facspy_adata = adata.copy()
    fp.tl.umap(facspy_adata,
               gate = "live",
               layer = "compensated",
               n_components = 2,
               random_state = 187,
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               n_comps = 12,
               n_neighbors = 10)
    np.testing.assert_array_almost_equal(scanpy_adata.obsm["X_pca"], facspy_adata.obsm["X_pca_live_compensated"])
    assert (scanpy_adata.obsp["connectivities"] != facspy_adata.obsp["live_compensated_neighbors_connectivities"]).nnz == 0
    
    assert np.array_equal(scanpy_adata.obsm["X_umap"],
                          facspy_adata.obsm["X_umap_live_compensated"])


def test_umap_same_as_scanpy_kwargs(mock_dataset: AnnData):
    """
    we first compute everything after one another
    and only change the umap call
    """
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    fp.tl.pca(scanpy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              exclude = None)
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    exclude = None)
    sc.tl.umap(scanpy_adata,
               n_components = 2,
               neighbors_key = "live_compensated_neighbors",
               random_state = 187)
    facspy_adata = adata.copy()
    fp.tl.pca(facspy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              exclude = None)
    fp.tl.neighbors(facspy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    exclude = None)
    fp.tl.umap(facspy_adata,
               gate = "live",
               layer = "compensated",
               n_components = 2,
               random_state = 187,
               use_only_fluo = False,
               exclude = None,
               scaling = None)
    np.testing.assert_array_almost_equal(scanpy_adata.obsm["X_pca_live_compensated"], facspy_adata.obsm["X_pca_live_compensated"])
    assert (scanpy_adata.obsp["live_compensated_neighbors_connectivities"] != facspy_adata.obsp["live_compensated_neighbors_connectivities"]).nnz == 0
    
    assert np.array_equal(scanpy_adata.obsm["X_umap"],
                          facspy_adata.obsm["X_umap_live_compensated"])

def test_umap_same_as_scanpy_no_precompute_pca_neighbors_kwargs(mock_dataset: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function
    """
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    scanpy_adata = adata.copy()
    scanpy_adata.X = scanpy_adata.layers["compensated"]
    fp.tl.pca(scanpy_adata,
              gate = "live",
              layer = "compensated",
              use_only_fluo = False,
              scaling = None,
              exclude = None)
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated",
                    use_only_fluo = False,
                    scaling = None,
                    exclude = None)
    sc.tl.umap(scanpy_adata,
               n_components = 2,
               neighbors_key = "live_compensated_neighbors",
               random_state = 187,
               min_dist = 1)
    facspy_adata = adata.copy()
    fp.tl.umap(facspy_adata,
               gate = "live",
               layer = "compensated",
               n_components = 2,
               random_state = 187,
               use_only_fluo = False,
               exclude = None,
               scaling = None,
               min_dist = 1)
    np.testing.assert_array_almost_equal(scanpy_adata.obsm["X_pca_live_compensated"], facspy_adata.obsm["X_pca_live_compensated"])
    assert (scanpy_adata.obsp["live_compensated_neighbors_connectivities"] != facspy_adata.obsp["live_compensated_neighbors_connectivities"]).nnz == 0
    
    assert np.array_equal(scanpy_adata.obsm["X_umap"],
                          facspy_adata.obsm["X_umap_live_compensated"])

def test_diffmap_same_as_scanpy(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    scanpy_adata = adata.copy()
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated")
    sc.tl.diffmap(scanpy_adata,
                  neighbors_key = "live_compensated_neighbors",
                  random_state = 187)
    
    facspy_adata = adata.copy()
    fp.tl.diffmap(facspy_adata,
                  gate = "live",
                  layer = "compensated",
                  random_state = 187)
    
    assert np.array_equal(scanpy_adata.obsm["X_diffmap"],
                          facspy_adata.obsm["X_diffmap_live_compensated"])

def test_diffmap_same_as_scanpy_kwargs(mock_dataset: AnnData):
    adata = mock_dataset
    fp.subset_gate(adata, "live")
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    scanpy_adata = adata.copy()
    fp.tl.neighbors(scanpy_adata,
                    gate = "live",
                    layer = "compensated")
    sc.tl.diffmap(scanpy_adata,
                  neighbors_key = "live_compensated_neighbors",
                  random_state = 187,
                  n_comps = 20)
    
    facspy_adata = adata.copy()
    # neighbors should be calculated internally
    fp.tl.diffmap(facspy_adata,
                  gate = "live",
                  layer = "compensated",
                  random_state = 187,
                  n_comps = 20)
    
    assert np.array_equal(scanpy_adata.obsm["X_diffmap"],
                          facspy_adata.obsm["X_diffmap_live_compensated"])


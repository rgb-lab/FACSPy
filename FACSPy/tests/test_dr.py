import pytest
import FACSPy as fp
import scanpy as sc
from anndata import AnnData
import numpy as np
from sklearn.manifold import TSNE

from FACSPy.exceptions._exceptions import InvalidScalingError

def test_invalid_scaling(mock_dataset_downsampled):
    with pytest.raises(InvalidScalingError):
        fp.tl.umap(mock_dataset_downsampled,
                   gate = "live",
                   scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.tsne(mock_dataset_downsampled,
                   gate = "live",
                   scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.pca(mock_dataset_downsampled,
                  gate = "live",
                  scaling = "MyCustomScaler")
    with pytest.raises(InvalidScalingError):
        fp.tl.diffmap(mock_dataset_downsampled,
                      gate = "live",
                      scaling = "MyCustomScaler")


def test_decorator_default_gate_and_default_layer_pca(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.pca(mock_dataset_downsampled)
    assert "X_pca_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_gate_provided_pca(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_layer = "compensated"

    fp.tl.pca(mock_dataset_downsampled, gate = "live")
    assert "X_pca_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_layer_provided_pca(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"

    fp.tl.pca(mock_dataset_downsampled, layer = "compensated")
    assert "X_pca_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_pca(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.pca(mock_dataset_downsampled)
    assert "X_pca_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg_pca(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.pca(mock_dataset_downsampled, "my_personal_gate")
    assert "X_pca_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg_pca(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.pca(mock_dataset_downsampled, gate = "my_personal_gate")
    assert "X_pca_live_compensated" in mock_dataset_downsampled.obsm


def test_pca_works_as_sklearn(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_pca_works_as_scanpy(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mock_dataset_downsampled.X = mock_dataset_downsampled.layers["compensated"]
    mock_dataset_downsampled = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    scanpy_adata = mock_dataset_downsampled.copy()
    facspy_adata = mock_dataset_downsampled.copy()
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

def test_pca_works_as_scanpy_2(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mock_dataset_downsampled.X = mock_dataset_downsampled.layers["compensated"]
    mock_dataset_downsampled = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    scanpy_adata = mock_dataset_downsampled.copy()
    facspy_adata = mock_dataset_downsampled.copy()
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
    
def test_pca_kwargs(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_pca_kwargs_2(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_decorator_default_gate_and_default_layer_tsne(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.tsne(mock_dataset_downsampled)
    assert "X_tsne_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_gate_provided_tsne(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_layer = "compensated"

    fp.tl.tsne(mock_dataset_downsampled, gate = "live")
    assert "X_tsne_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_layer_provided_tsne(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"

    fp.tl.tsne(mock_dataset_downsampled, layer = "compensated")
    assert "X_tsne_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_tsne(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.tsne(mock_dataset_downsampled)
    assert "X_tsne_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg_tsne(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.tsne(mock_dataset_downsampled, "my_personal_gate")
    assert "X_tsne_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg_tsne(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.tsne(mock_dataset_downsampled, gate = "my_personal_gate")
    assert "X_tsne_live_compensated" in mock_dataset_downsampled.obsm

def test_tsne_works_as_sklearn(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 early_exaggeration = 12,
                 learning_rate = "auto",
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
    assert uns_dict["params"]["learning_rate"] == "auto"
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["use_rep"] == "X"
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["n_components"] == 2

    assert adata.X is None


def test_tsne_works_as_scanpy(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mock_dataset_downsampled.X = mock_dataset_downsampled.layers["compensated"]
    mock_dataset_downsampled = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    scanpy_adata = mock_dataset_downsampled.copy()
    facspy_adata = mock_dataset_downsampled.copy()
    sc.tl.tsne(scanpy_adata,
               random_state = 187,
               use_rep = "X",
               learning_rate = "auto")
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

def test_tsne_works_as_scanpy_2(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    mock_dataset_downsampled.X = mock_dataset_downsampled.layers["compensated"]
    mock_dataset_downsampled = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    scanpy_adata = mock_dataset_downsampled.copy()
    facspy_adata = mock_dataset_downsampled.copy()
    sc.tl.tsne(scanpy_adata,
               early_exaggeration = 400,
               use_rep = "X",
               learning_rate = "auto",
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
    assert uns_dict["params"]["learning_rate"] == "auto"
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["use_rep"] == "X"
    assert uns_dict["params"]["n_components"] == 2
    assert facspy_adata.X is None
    assert np.array_equal(scanpy_adata.obsm["X_tsne"],
                          facspy_adata.obsm["X_tsne_live_compensated"])

def test_tsne_works_as_sklearn_with_rep(mock_dataset_downsampled: AnnData):
    """tests if rep pca works"""
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_tsne_works_as_sklearn_with_rep_2(mock_dataset_downsampled: AnnData):
    """tests if the npcs parameter works"""
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_tsne_choose_rep_value_error(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    fp.tl.pca(adata,
              gate = "live",
              layer = "compensated")
    with pytest.raises(ValueError):
        fp.tl.tsne(adata,
                   gate = "live",
                   layer = "compensated",
                   use_rep = "X_pca_live_compensated",
                   n_pcs = 30)
        
def test_tsne_rep_does_not_exist(mock_dataset_downsampled: AnnData):
    adata = mock_dataset_downsampled.copy()
    with pytest.raises(ValueError):
        fp.tl.tsne(adata,
                   gate = "live",
                   layer = "compensated",
                   use_rep = "X_pca_live_compensated",
                   n_pcs = 30)

def test_tsne_kwargs(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    test_tsne_ = TSNE(n_components = 2,
                      perplexity = 30,
                      early_exaggeration = 12,
                      learning_rate = 1000,
                      random_state = 187)
    test_coords = test_tsne_.fit_transform(adata.layers["compensated"])

    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 early_exaggeration = 400,
                 learning_rate = "auto",
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
    assert uns_dict["params"]["learning_rate"] == "auto"
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "euclidean"
    assert uns_dict["params"]["n_components"] == 2

    assert adata.X is None

def test_tsne_kwargs_2(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
    test_tsne_ = TSNE(n_components = 2,
                      perplexity = 30,
                      metric = "euclidean",
                      early_exaggeration = 12,
                      learning_rate = "auto",
                      random_state = 187)
    test_coords = test_tsne_.fit_transform(adata.layers["compensated"])

    tsne_ = TSNE(n_components = 2,
                 perplexity = 30,
                 metric = "cosine",
                 early_exaggeration = 12,
                 learning_rate = "auto",
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
    assert uns_dict["params"]["learning_rate"] == "auto"
    assert uns_dict["params"]["n_jobs"] == 1
    assert uns_dict["params"]["metric"] == "cosine"
    assert uns_dict["params"]["n_components"] == 2
    assert uns_dict["params"]["use_rep"] == "X"

    assert adata.X is None

def test_decorator_default_gate_and_default_layer_umap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.umap(mock_dataset_downsampled)
    assert "X_umap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_gate_provided_umap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_layer = "compensated"

    fp.tl.umap(mock_dataset_downsampled, gate = "live")
    assert "X_umap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_layer_provided_umap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"

    fp.tl.umap(mock_dataset_downsampled, layer = "compensated")
    assert "X_umap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_umap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.umap(mock_dataset_downsampled)
    assert "X_umap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg_umap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.umap(mock_dataset_downsampled, "my_personal_gate")
    assert "X_umap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg_umap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.umap(mock_dataset_downsampled, gate = "my_personal_gate")
    assert "X_umap_live_compensated" in mock_dataset_downsampled.obsm

def test_umap_same_as_scanpy(mock_dataset_downsampled: AnnData):
    """
    we first compute everything after one another
    and only change the umap call
    """
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_umap_same_as_scanpy_no_precompute_pca_neighbors(mock_dataset_downsampled: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function
    """
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_umap_same_as_scanpy_no_precompute_pca_neighbors_scanpy_funcs(mock_dataset_downsampled: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function and compare this to the scanpy
    pca and neighbors call
    """
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_umap_same_as_scanpy_no_precompute_pca_neighbors_scanpy_funcs_kwargs(mock_dataset_downsampled: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function and compare this to the scanpy
    pca and neighbors call
    """
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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


def test_umap_same_as_scanpy_kwargs(mock_dataset_downsampled: AnnData):
    """
    we first compute everything after one another
    and only change the umap call
    """
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_umap_same_as_scanpy_no_precompute_pca_neighbors_kwargs(mock_dataset_downsampled: AnnData):
    """
    now we see if the pca and neighbors get computed correctly
    within the umap function
    """
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_decorator_default_gate_and_default_layer_dmap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"

    fp.tl.diffmap(mock_dataset_downsampled)
    assert "X_diffmap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_gate_provided_dmap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_layer = "compensated"

    fp.tl.diffmap(mock_dataset_downsampled, gate = "live")
    assert "X_diffmap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_only_layer_provided_dmap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"

    fp.tl.diffmap(mock_dataset_downsampled, layer = "compensated")
    assert "X_diffmap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_dmap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.diffmap(mock_dataset_downsampled)
    assert "X_diffmap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_arg_dmap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.diffmap(mock_dataset_downsampled, "my_personal_gate")
    assert "X_diffmap_live_compensated" in mock_dataset_downsampled.obsm

def test_decorator_default_gate_and_default_layer_and_gate_alias_use_alias_as_kwarg_dmap(mock_dataset_downsampled: AnnData):
    mock_dataset_downsampled = mock_dataset_downsampled.copy()
    fp.settings.default_gate = "live"
    fp.settings.default_layer = "compensated"
    fp.settings.add_new_alias("live", "my_personal_gate")

    fp.tl.diffmap(mock_dataset_downsampled, gate = "my_personal_gate")
    assert "X_diffmap_live_compensated" in mock_dataset_downsampled.obsm

def test_diffmap_same_as_scanpy(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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

def test_diffmap_same_as_scanpy_kwargs(mock_dataset_downsampled: AnnData):
    adata = fp.subset_gate(mock_dataset_downsampled, "live", copy = True)
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
